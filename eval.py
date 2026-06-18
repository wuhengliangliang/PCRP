# eval.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', message='Better speed can be achieved with apex installed')

import os
import sys
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import evaluate
from models import build_model
from datasets import build_dataset, test_transforms
from util.logger import get_logger
from util.config import Config


# =========================
# ✅ 保证 sam3 能 import（与你 train.py 一致，sam3/sam3 结构）
# =========================
_THIS_DIR = Path(__file__).resolve().parent
_SAM3_PARENT = _THIS_DIR / "sam3"   # PathVG-main/sam3
if _SAM3_PARENT.exists():
    sp = str(_SAM3_PARENT)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def boolean_string(s):
    if isinstance(s, bool):
        return s
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# =========================
# ✅ PyTorch 2.6+ 兼容 torch.load(weights_only 默认值变化)
# =========================
def safe_torch_load(path, map_location="cpu"):
    """
    PyTorch 2.6+ 默认 weights_only=True，会导致包含 Namespace / rng_state 等对象的 ckpt 无法加载。
    如果 ckpt 是你自己训练保存的（可信来源），强制 weights_only=False 即可恢复兼容。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_device(args) -> torch.device:
    """
    ✅ DDP 下每个 rank 必须用自己的 cuda:{gpu}
    """
    dev = getattr(args, "device", "cuda")
    if isinstance(dev, str) and (dev == "cuda" or dev.startswith("cuda")):
        if torch.cuda.is_available():
            if getattr(args, "distributed", False):
                return torch.device(f"cuda:{args.gpu}")
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(dev)


def _ensure_sam3_alias(args):
    """
    兼容不同命名：
    - config: sam3_bpe_path
    - 有些代码可能用 args.sam3_bpe
    """
    if hasattr(args, "sam3_bpe_path") and (not getattr(args, "sam3_bpe", "")):
        args.sam3_bpe = getattr(args, "sam3_bpe_path", "")


# =========================
# ✅ 支持目录型/分片 checkpoint
#   ckpt_dir/
#     meta.pth   (包含 shards 列表)
#     rank000.pth / shard000.pth ...
# =========================
def load_sharded_checkpoint_dir(ckpt_dir: str, map_location="cpu") -> Dict[str, Any]:
    ckpt_dir_p = Path(ckpt_dir)
    meta_path = ckpt_dir_p / "meta.pth"
    if not meta_path.exists():
        raise FileNotFoundError(f"[ShardedCKPT] meta.pth not found under: {ckpt_dir}")

    meta = safe_torch_load(str(meta_path), map_location=map_location)
    if not isinstance(meta, dict):
        raise RuntimeError(f"[ShardedCKPT] meta.pth is not a dict: {ckpt_dir}")

    shard_files = meta.get("shards", [])
    if not isinstance(shard_files, list) or len(shard_files) == 0:
        raise RuntimeError(f"[ShardedCKPT] meta['shards'] invalid or empty in: {ckpt_dir}")

    full_sd: Dict[str, Any] = {}
    for fn in shard_files:
        fp = ckpt_dir_p / str(fn)
        if not fp.exists():
            raise FileNotFoundError(f"[ShardedCKPT] shard missing: {fp}")
        part = safe_torch_load(str(fp), map_location=map_location)
        if isinstance(part, dict) and ("model" in part) and isinstance(part["model"], dict):
            full_sd.update(part["model"])
        elif isinstance(part, dict):
            # 兼容某些实现 shard 就是 state_dict
            full_sd.update(part)
        else:
            raise RuntimeError(f"[ShardedCKPT] shard format not supported: {fp}")

    meta["model"] = full_sd
    return meta


def load_checkpoint_any(path: str, map_location="cpu") -> Dict[str, Any]:
    """
    统一入口：
    - 文件：torch.load
    - 目录：meta.pth + shards 合并
    """
    p = Path(path)
    if p.is_dir():
        return load_sharded_checkpoint_dir(str(p), map_location=map_location)
    return safe_torch_load(str(p), map_location=map_location)


def get_args_parser():
    parser = argparse.ArgumentParser('TransCP: Evaluation only', add_help=False)

    # ========== base ==========
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--seed', default=3407, type=int)

    # ========== core (must exist for Config.merge_to_args) ==========
    parser.add_argument('--dataset', default='pathology2', type=str)
    parser.add_argument('--output_dir', default='outputs/pathology_reason_test/public', type=str)
    parser.add_argument('--checkpoint_best', action='store_true')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--freeze_epochs', default=10, type=int)
    parser.add_argument('--freeze_modules', nargs='+', default=['backbone'], type=str)
    parser.add_argument('--load_weights_path', default='pretrained_checkpoints/detr-r50.pth', type=str)

    # config 中 model_config 是 dict：保持可 eval（仅 CLI 用，cfg.merge_to_args 会直接覆盖为 dict）
    parser.add_argument('--model_config', default=None, type=eval)

    # ========== optimizer/lr args（build_model/optimizer 依赖，评估时也需要 args 字段存在） ==========
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_losses', type=list, default=[])

    # ========== model params ==========
    parser.add_argument('--resume',
                        default='/data_3/pl/miccai/pathvgsam3/outputs/pathology_reason_test/sam3_new/ckpts/last/meta.pth',
                        help='resume from checkpoint (file or sharded dir)')
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_path', default='pretrained_checkpoints/resnet50-19c8e357.pth', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))

    # * Transformer (visual encoder)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=1, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # * Bert (language encoder)
    parser.add_argument('--bert_model', default='pretrained_checkpoints/bert-base-uncased.tar.gz', type=str)
    parser.add_argument('--bert_token_mode', default='pretrained_checkpoints/bert_base_uncased', type=str)
    parser.add_argument('--bert_output_dim', default=768, type=int)
    parser.add_argument('--bert_output_layers', default=12, type=int)
    parser.add_argument('--max_query_len', default=40, type=int)
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # ========== loss ==========
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--loss_loc', default='loss_boxes', type=str)
    parser.add_argument('--box_xyxy', action='store_true')
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # ========== dataset ==========
    parser.add_argument('--data_root', default='/data_3/pl/miccai/refpath_image')
    parser.add_argument('--split_root', default='/data_3/pl/miccai/PathVG-main/split/data')
    parser.add_argument('--test_split', default='testB')
    parser.add_argument('--img_size', default=640)
    parser.add_argument('--cache_images', action='store_true')
    parser.add_argument('--save_pred_path', default='predictions.json', help='path to save prediction results')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--batch_size_test', default=16, type=int)
    parser.add_argument('--test_transforms', default=test_transforms)

    # ========== distributed ==========
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # ========== config ==========
    parser.add_argument('--config', default='configs/TransCP_R50_pathology2.py', type=str)

    # ========== ✅ SAM3 teacher-in-the-loop (与你 config 对齐) ==========
    parser.add_argument('--use_sam3', default=True, type=boolean_string)
    parser.add_argument('--sam3_ckpt', default='', type=str)
    parser.add_argument('--sam3_bpe_path', default='', type=str)
    parser.add_argument('--sam3_resolution', default=1008, type=int)
    parser.add_argument('--sam3_confidence_threshold', default=0.9, type=float)
    parser.add_argument('--sam3_prompt_coord', default='norm', type=str, choices=['norm', 'pixel'])
    parser.add_argument('--sam3_bpe', default='', type=str)

    # ========== debug ==========
    parser.add_argument('--strict_load', default=False, type=boolean_string,
                        help='If True, load_state_dict(strict=True) to hard fail on mismatch.')
    parser.add_argument('--print_ckpt_summary', default=True, type=boolean_string,
                        help='Print ckpt meta fields like epoch/best_acc when available.')

    return parser


def _log_ckpt_summary(logger, ckpt: Dict[str, Any]):
    if not isinstance(ckpt, dict):
        return
    keys = list(ckpt.keys())
    logger.info(f"[CKPT] top-level keys ({len(keys)}): {keys[:30]}")
    for k in ["epoch", "best_acc", "best_metric", "tag", "uuid", "world_size", "shard_by_rank"]:
        if k in ckpt:
            logger.info(f"[CKPT] {k} = {ckpt.get(k)}")


def _load_model_weights(logger, model_wo_ddp, resume_path: str, strict_load: bool = False):
    ckpt = load_checkpoint_any(resume_path, map_location="cpu")

    if isinstance(ckpt, dict) and ("model" in ckpt) and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        # 兼容直接 state_dict
        sd = ckpt
    else:
        raise RuntimeError(f"[CKPT] Unsupported checkpoint type: {type(ckpt)}")

    missing, unexpected = model_wo_ddp.load_state_dict(sd, strict=bool(strict_load))
    logger.info(f"[CKPT] Loaded weights from: {resume_path}")
    logger.info(f"[CKPT] strict_load={strict_load}")
    logger.info(f"[CKPT] Missing keys: {len(missing)} (first 40) {missing[:40]}")
    logger.info(f"[CKPT] Unexpected keys: {len(unexpected)} (first 40) {unexpected[:40]}")
    return ckpt


def main(args):
    # init distributed
    utils.init_distributed_mode(args)
    if getattr(args, "distributed", False) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # logger
    eval_output_dir = Path(args.output_dir) / "eval_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluation", eval_output_dir, utils.get_rank(), filename='eval.log')
    logger.info("===== TransCP Evaluation Mode (Debug) =====")
    logger.info(f"Loaded config from: {args.config}")
    logger.info(args)

    # device
    device = _resolve_device(args)
    if (str(device).startswith("cuda")) and (not torch.cuda.is_available()):
        logger.warning("CUDA is not available, switching to CPU")
        device = torch.device('cpu')
    logger.info(f"[Device] Using device = {device}")

    # seed
    seed = int(args.seed) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # sam3 alias
    _ensure_sam3_alias(args)

    # build
    model, criterion, postprocessor = build_model(args)
    model.to(device)
    logger.info(f"Model built. use_sam3={getattr(args,'use_sam3', False)} | model_config type={type(args.model_config)}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # load ckpt
    if args.resume:
        ckpt = _load_model_weights(
            logger, model_without_ddp, args.resume, strict_load=bool(args.strict_load)
        )
        if bool(getattr(args, "print_ckpt_summary", True)):
            _log_ckpt_summary(logger, ckpt)
    elif args.load_weights_path:
        model_without_ddp.load_pretrained_weights(args.load_weights_path)
        logger.info(f"Loaded pretrained weights from: {args.load_weights_path}")
    else:
        logger.warning("No model weights loaded! Using random initialization.")

    # dataset
    dataset_test = build_dataset(test=True, args=args)
    logger.info(f"Test dataset ({args.dataset}/{args.test_split}) size: {len(dataset_test)}")

    # loader
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=int(args.batch_size_test),
        sampler=sampler_test,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=int(args.num_workers)
    )

    # eval
    logger.info("Starting evaluation...")
    start_time = time.time()

    save_path = str(eval_output_dir / args.save_pred_path) if args.save_pred_path else ""

    # ✅ 尽量与 train.py 调用一致：优先传 args；若 evaluate 不支持则 fallback
    try:
        test_stats, test_acc, test_time = evaluate(
            model, criterion, postprocessor, data_loader_test, device, save_path,
            args=args, writer=None, global_step=0, tb_num_vis=0, tb_tag_prefix="eval"
        )
    except TypeError:
        test_stats, test_acc, test_time = evaluate(
            model, criterion, postprocessor, data_loader_test, device, save_path
        )

    eval_time = time.time() - start_time

    logger.info("===== Evaluation Results =====")
    if isinstance(test_stats, dict):
        logger.info('Loss Stats | ' + ' | '.join([f'{k}: {v:.4f}' for k, v in test_stats.items()]))
    else:
        logger.info(f"Loss Stats | {test_stats}")

    if isinstance(test_acc, dict):
        logger.info('Accuracy   | ' + ' | '.join([f'{k}: {v:.4f}' for k, v in test_acc.items()]))
    else:
        logger.info(f"Accuracy   | {test_acc}")

    logger.info(f"Total evaluation time: {datetime.timedelta(seconds=int(eval_time))}")
    logger.info(f"Test time details: {test_time}")

    if utils.is_main_process():
        results = {
            'dataset': args.dataset,
            'test_split': args.test_split,
            'model_checkpoint': args.resume,
            'use_sam3': getattr(args, "use_sam3", False),
            'sam3_ckpt': getattr(args, "sam3_ckpt", ""),
            'sam3_bpe_path': getattr(args, "sam3_bpe_path", ""),
            'sam3_resolution': getattr(args, "sam3_resolution", None),
            'sam3_confidence_threshold': getattr(args, "sam3_confidence_threshold", None),
            'sam3_prompt_coord': getattr(args, "sam3_prompt_coord", None),
            'test_stats': test_stats,
            'test_acc': test_acc,
            'test_time': test_time,
            'total_eval_time': str(datetime.timedelta(seconds=int(eval_time))),
        }
        with open(eval_output_dir / 'eval_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to: {eval_output_dir / 'eval_results.json'}")

    # clean DDP
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransCP Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()

    # load config
    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)  # ✅ sam3_* 字段不会 assert

    main(args)
