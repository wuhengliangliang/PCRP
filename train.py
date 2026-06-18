# train.py
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', message='Better speed can be achieved with apex installed')

import os
import sys
import argparse
import datetime
import random
import time
import tempfile
import shutil
import json
import uuid
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import train_one_epoch, train_one_epoch_w_accum, evaluate
from models import build_model
from datasets import build_dataset, train_transforms, test_transforms
from util.logger import get_logger
from util.config import Config


# =========================
# ensure sam3 importable (if repo has sam3/)
# =========================
_THIS_DIR = Path(__file__).resolve().parent
_SAM3_PARENT = _THIS_DIR / "sam3"
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


def _get_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter  # type: ignore
            return SummaryWriter
        except Exception:
            return None


SummaryWriter = _get_summary_writer()


# =========================
# ✅ STDOUT/STDERR Tee -> iter.txt (captures non-logger prints)
# =========================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _install_iter_tee(output_dir: Path, rank: int):
    """
    将所有 print/stdout/stderr 复制到 iter.txt（主进程）
    其他 rank 写到 iter_rankXXX.txt，避免互相覆盖。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        fpath = output_dir / "iter.txt"
    else:
        fpath = output_dir / f"iter_rank{rank:03d}.txt"
    f = open(str(fpath), "a", buffering=1, encoding="utf-8", errors="ignore")

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, f)
    sys.stderr = _Tee(old_err, f)
    return f, old_out, old_err


def safe_torch_load(path, map_location="cpu"):
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    except RuntimeError as e:
        msg = str(e)
        if ("PytorchStreamReader" in msg) or ("failed finding central directory" in msg):
            raise RuntimeError(
                f"[Resume] Checkpoint seems CORRUPTED (half-written): {path}\n"
                f"Reason: {msg}\n"
                f"Fix: resume from best/last epoch ckpt, and enable atomic saving."
            ) from e
        raise


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


# -------------------------
# Atomic checkpoint saving
# -------------------------
def _atomic_torch_save(obj, path: Path, logger=None, fsync_dir: bool = False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)

    try:
        if logger is not None:
            logger.info(f"[CKPT] torch.save -> tmp start: {Path(tmp_path).name}")
        t0 = time.time()

        try:
            torch.save(obj, tmp_path, _use_new_zipfile_serialization=True)
        except TypeError:
            torch.save(obj, tmp_path)

        t1 = time.time()
        if logger is not None:
            try:
                sz = os.path.getsize(tmp_path) / (1024 ** 2)
                logger.info(f"[CKPT] torch.save tmp done: {t1 - t0:.2f}s, size={sz:.1f}MB")
            except Exception:
                logger.info(f"[CKPT] torch.save tmp done: {t1 - t0:.2f}s")

        os.replace(tmp_path, str(path))

        if fsync_dir:
            try:
                dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
                os.fsync(dir_fd)
                os.close(dir_fd)
            except Exception:
                pass

        if logger is not None:
            logger.info(f"[CKPT] os.replace done -> {path.name}")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# Sharded + Async Checkpoint
# =========================
def _to_cpu_state_dict(sd: Dict[str, Any], force_fp32: bool = False) -> Dict[str, Any]:
    """
    ✅ 保存 ckpt 的 CPU snapshot：
      - 默认：仅搬到 cpu（保持原 dtype）
      - force_fp32=True：所有浮点张量统一转成 float32 再保存（fp16/bf16/fp64 -> fp32）
    """
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            t = v.detach().to("cpu")
            if bool(force_fp32) and t.is_floating_point() and t.dtype != torch.float32:
                t = t.float()
            out[k] = t
        else:
            out[k] = v
    return out


def _split_keys_even(keys: List[str], num_shards: int) -> List[List[str]]:
    num_shards = max(1, int(num_shards))
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    for i, k in enumerate(keys):
        shards[i % num_shards].append(k)
    return shards


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _wait_for_files(files: List[Path], timeout_s: float = 600.0, sleep_s: float = 0.2) -> bool:
    t0 = time.time()
    while True:
        ok = True
        for f in files:
            if not f.exists():
                ok = False
                break
        if ok:
            return True
        if (time.time() - t0) > timeout_s:
            return False
        time.sleep(sleep_s)


def _save_sharded_checkpoint_dir(
    *,
    ckpt_dir: Path,
    tag: str,
    model_sd_cpu: Optional[Dict[str, Any]],
    meta_payload: Optional[Dict[str, Any]],
    shard_filenames: List[str],
    logger=None,
    fsync_dir: bool = False,
):
    _ensure_dir(ckpt_dir)

    if model_sd_cpu is not None:
        shard_name = shard_filenames[0]
        shard_path = ckpt_dir / shard_name
        _atomic_torch_save({"model": model_sd_cpu}, shard_path, logger=logger, fsync_dir=fsync_dir)

    if meta_payload is not None:
        shard_list = meta_payload.get("shards", [])
        shard_paths = [ckpt_dir / str(x) for x in shard_list] if isinstance(shard_list, list) else []
        if shard_paths:
            ok = _wait_for_files(shard_paths, timeout_s=float(meta_payload.get("meta_wait_timeout_s", 600.0)))
            if (logger is not None) and (not ok):
                logger.info(f"[CKPT][{tag}] meta wait shards TIMEOUT (still write meta): {ckpt_dir}")

        meta_path = ckpt_dir / "meta.pth"
        _atomic_torch_save(meta_payload, meta_path, logger=logger, fsync_dir=fsync_dir)


def load_sharded_checkpoint_dir(ckpt_dir: str, map_location="cpu"):
    ckpt_dir_p = Path(ckpt_dir)
    meta = safe_torch_load(ckpt_dir_p / "meta.pth", map_location=map_location)
    shard_files = meta.get("shards", [])
    full_sd: Dict[str, Any] = {}
    if isinstance(shard_files, list):
        for fn in shard_files:
            part = safe_torch_load(ckpt_dir_p / fn, map_location=map_location)
            if isinstance(part, dict) and ("model" in part):
                full_sd.update(part["model"])
    meta["model"] = full_sd
    return meta


class AsyncShardedCheckpointer:
    def __init__(self, logger=None, fsync_dir: bool = False, max_pending: int = 1):
        self.logger = logger
        self.fsync_dir = bool(fsync_dir)
        self.q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(max_pending)))
        self._stop = object()
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def submit(self, job: dict):
        try:
            while self.q.full():
                _ = self.q.get_nowait()
                self.q.task_done()
        except Exception:
            pass
        self.q.put(job)

    def close(self, wait: bool = True):
        try:
            self.q.put(self._stop)
        except Exception:
            pass
        if wait:
            try:
                self.th.join()
            except Exception:
                pass

    def _worker(self):
        while True:
            job = self.q.get()
            if job is self._stop:
                self.q.task_done()
                break
            try:
                _save_sharded_checkpoint_dir(**job)
            except Exception as e:
                if self.logger is not None:
                    self.logger.info(f"[CKPT][Async] save failed: {repr(e)}")
            finally:
                self.q.task_done()


# -------------------------
# (Legacy) GLOO barrier for ckpt sync
# -------------------------
def build_ckpt_sync_group(logger, hours: int = 12):
    if not is_dist_ready():
        return None
    try:
        return dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=int(hours)))
    except Exception as e:
        if logger is not None:
            logger.info(f"[CKPT] build gloo sync group failed: {repr(e)}; fallback to default barrier.")
        return "default"


def ckpt_barrier(group, logger=None, tag: str = ""):
    if not is_dist_ready() or group is None:
        return
    try:
        if logger is not None and tag:
            logger.info(f"[CKPT] barrier enter {tag}")
        if group == "default":
            dist.barrier()
        else:
            dist.barrier(group=group)
        if logger is not None and tag:
            logger.info(f"[CKPT] barrier exit  {tag}")
    except Exception:
        try:
            dist.barrier()
        except Exception:
            pass


def optimizer_state_to_cpu(optim_state):
    if not isinstance(optim_state, dict):
        return optim_state

    def _to_cpu(x):
        if torch.is_tensor(x):
            return x.detach().to("cpu")
        if isinstance(x, dict):
            return {kk: _to_cpu(vv) for kk, vv in x.items()}
        if isinstance(x, (list, tuple)):
            t = [_to_cpu(vv) for vv in x]
            return type(x)(t)
        return x

    return _to_cpu(optim_state)


def _get_rng_state():
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict):
    try:
        if "python" in st:
            random.setstate(st["python"])
        if "numpy" in st:
            np.random.set_state(st["numpy"])
        if "torch" in st:
            torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and ("cuda" in st):
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception:
        pass


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def _load_ckpt(
    path: str,
    model_wo_ddp,
    optimizer,
    lr_scheduler,
    args,
    logger,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    ✅ 返回 (best_testA, best_testB, best_joint)，兼容旧 ckpt 的 best_acc
    """
    if Path(path).is_dir():
        ckpt = load_sharded_checkpoint_dir(path, map_location="cpu")
    else:
        ckpt = safe_torch_load(path, map_location="cpu")

    state = ckpt.get("model", ckpt)
    missing, unexpected = model_wo_ddp.load_state_dict(state, strict=False)
    logger.info(f"[Resume] Loaded model weights from: {path}")
    if len(missing) > 0:
        logger.info(f"[Resume] Missing keys: {len(missing)} (first 20) {missing[:20]}")
    if len(unexpected) > 0:
        logger.info(f"[Resume] Unexpected keys: {len(unexpected)} (first 20) {unexpected[:20]}")

    legacy_best = float(ckpt.get("best_acc", 0.0))
    best_testA = float(ckpt.get("best_testA", legacy_best))
    best_testB = float(ckpt.get("best_testB", legacy_best))
    best_joint = float(ckpt.get("best_joint", legacy_best))

    if "epoch" in ckpt:
        args.start_epoch = int(ckpt["epoch"]) + 1
        logger.info(f"[Resume] Found epoch={ckpt['epoch']}. Set start_epoch={args.start_epoch}")

    if (not args.eval) and ("optimizer" in ckpt) and ("lr_scheduler" in ckpt):
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            _move_optimizer_state_to_device(optimizer, device)
            logger.info("[Resume] Loaded optimizer & lr_scheduler (moved optimizer state to device).")
        except Exception as e:
            logger.info(f"[Resume] Optimizer/scheduler load failed: {e}")
    else:
        logger.info("[Resume] Optimizer/lr_scheduler not found in ckpt (model-only resume).")

    if "rng_state" in ckpt:
        _set_rng_state(ckpt["rng_state"])
        logger.info("[Resume] Restored RNG state.")

    logger.info(f"[Resume] best_testA={best_testA:.6f} best_testB={best_testB:.6f} best_joint={best_joint:.6f}")
    return best_testA, best_testB, best_joint


def _pick_score(d: dict) -> float:
    if not isinstance(d, dict) or len(d) == 0:
        return 0.0
    for k in ["acc", "accuracy", "mAP", "map", "AP", "iou", "IoU", "success", "SR", "Recall", "Mean_iou", "Acc@0.50"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    try:
        return float(next(iter(d.values())))
    except Exception:
        return 0.0


def _pick_acc50(d: dict) -> float:
    if not isinstance(d, dict) or len(d) == 0:
        return 0.0
    for k in ["Acc@0.50", "Acc@0.5", "acc@0.50", "acc@0.5"]:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                return 0.0
    return _pick_score(d)


def _pick_acc_at(d: dict, thr: float) -> float:
    """
    从 val_acc dict 里取 Acc@thr（比如 0.70），取不到就回退到 _pick_score。
    """
    if not isinstance(d, dict) or len(d) == 0:
        return 0.0
    keys = [
        f"Acc@{thr:.2f}", f"Acc@{thr:.1f}",
        f"acc@{thr:.2f}", f"acc@{thr:.1f}",
        f"Acc@{thr}", f"acc@{thr}",
    ]
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                return 0.0
    return _pick_score(d)


def _resolve_device(args) -> torch.device:
    dev = getattr(args, "device", "cuda")
    if isinstance(dev, str) and (dev == "cuda" or dev.startswith("cuda")):
        if torch.cuda.is_available():
            if getattr(args, "distributed", False):
                return torch.device(f"cuda:{args.gpu}")
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(dev)


def _parse_keywords(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def filter_state_dict(sd: Dict[str, Any], keywords_lower: list):
    if not keywords_lower:
        return sd
    out = {}
    for k, v in sd.items():
        kl = k.lower()
        if any(kw in kl for kw in keywords_lower):
            continue
        out[k] = v
    return out


def sanitize_args_for_ckpt(args) -> Dict[str, Any]:
    d = vars(args).copy()

    def _safe(x):
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, (list, tuple)):
            return [_safe(v) for v in x]
        if isinstance(x, dict):
            return {str(kk): _safe(vv) for kk, vv in x.items()}

        try:
            name = getattr(x, "__name__", None)
            if name is not None and isinstance(name, str):
                return f"<fn:{name}>"
        except Exception:
            pass
        return repr(x)

    return {k: _safe(v) for k, v in d.items()}


# =========================
# ✅ 关键修复：不漏参数的 optimizer 分组 + 打印
# =========================
def build_param_groups(model, args, logger=None):
    """
    遍历所有 named_parameters，按名字分 lr，不漏任何 requires_grad=True 的参数。
    同时排除 sam3/teacher（teacher 不训练）。
    """
    base_lr = float(args.lr)
    wd = float(args.weight_decay)

    exclude_kw = ["sam3", "teacher", "inst_interactive_predictor"]
    text_kw = ["bert", "textmodel", "lstm", "language", "transcp_text"]

    groups = {}   # lr -> dict(params, lr, weight_decay, names)
    seen = set()

    def _get_group(lr: float):
        k = float(lr)
        if k not in groups:
            groups[k] = {"params": [], "lr": float(lr), "weight_decay": wd, "names": []}
        return groups[k]

    total = 0
    skipped = 0

    for n, p in model.named_parameters():
        total += 1
        if p is None or (not p.requires_grad):
            continue

        nl = n.lower()
        if any(kw in nl for kw in exclude_kw):
            skipped += 1
            continue

        # lr 规则
        if nl.startswith("backbone."):
            lr = float(args.lr_backbone)
        elif nl.startswith("trans_encoder.") or nl.startswith("input_proj."):
            lr = float(args.lr_vis_enc)
        elif any(kw in nl for kw in text_kw):
            lr = float(args.lr_bert)
        else:
            lr = base_lr

        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)

        g = _get_group(lr)
        g["params"].append(p)
        g["names"].append(n)

    param_groups = list(groups.values())

    if logger is not None:
        logger.info("[OPT] ===== Param groups summary (NO-MISS) =====")
        for g in sorted(param_groups, key=lambda x: x["lr"]):
            names = g.get("names", [])
            logger.info(f"[OPT] lr={g['lr']:.2e}  n_params={len(names)}  eg={names[:8]}")
        logger.info(f"[OPT] total_named_params={total}, excluded_by_kw={skipped}, optimized_unique={len(seen)}")

        # 关键模块是否存在（基于 state_dict key）
        sd_keys = list(model.state_dict().keys())

        def _has_any(subs: List[str]) -> bool:
            subs = [s.lower() for s in subs]
            for k in sd_keys:
                kl = k.lower()
                if any(s in kl for s in subs):
                    return True
            return False

        logger.info(f"[OPT] has(mask_prior_adapter)={_has_any(['mask_prior', 'mask_adapter'])}")
        logger.info(f"[OPT] has(textmodel/lstm)={_has_any(['textmodel', 'lstm', 'transcp_text'])}")
        logger.info(f"[OPT] has(prompt_fusion/gate)={_has_any(['fusion', 'gate', 'prompt'])}")

        # requires_grad 计数（更直接判断“有没有被冻住”）
        def _cnt_req(subs: List[str]) -> int:
            subs = [s.lower() for s in subs]
            c = 0
            for nn, pp in model.named_parameters():
                if pp is None:
                    continue
                nl = nn.lower()
                if any(s in nl for s in subs) and pp.requires_grad:
                    c += 1
            return c

        logger.info(f"[GRAD] requires_grad count mask_prior={_cnt_req(['mask_prior', 'mask_adapter'])}")
        logger.info(f"[GRAD] requires_grad count textmodel={_cnt_req(['textmodel', 'lstm', 'transcp_text', 'bert'])}")
        logger.info(f"[GRAD] requires_grad count fusion/gate={_cnt_req(['fusion', 'gate', 'prompt'])}")

    return param_groups


def get_args_parser():
    parser = argparse.ArgumentParser('TransCP training script', add_help=False)
    parser.add_argument('--local_rank', default=0, type=int)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--amp', default=True, type=boolean_string)
    parser.add_argument('--use_prefetcher', default=False, type=boolean_string)
    parser.add_argument('--accum_iter', default=2, type=int)

    parser.add_argument('--checkpoint_step', default=50, type=int)
    parser.add_argument('--checkpoint_latest', action='store_true')
    parser.add_argument('--checkpoint_best', action='store_true')

    # ✅ 修改：默认保存最后 3 个 epoch 的 ckpt（epoch_XXXX）
    parser.add_argument('--save_last_k', default=3, type=int)

    parser.add_argument('--save_optimizer_state', default=False, type=boolean_string)
    parser.add_argument('--save_latest_with_optim', default=False, type=boolean_string)
    parser.add_argument('--best_save_interval', default=5, type=int)
    parser.add_argument('--ckpt_fsync_dir', default=False, type=boolean_string)
    parser.add_argument('--ckpt_sync', default=False, type=boolean_string)
    parser.add_argument('--ckpt_sync_timeout_hours', default=12, type=int)

    parser.add_argument('--ckpt_filter_keywords', default="sam3,teacher", type=str)
    parser.add_argument('--ckpt_filter_enable', default=True, type=boolean_string)

    # ✅ float32那个：保存 ckpt CPU snapshot 时强制 float32
    parser.add_argument('--ckpt_force_cpu_snapshot', default=False, type=boolean_string)

    parser.add_argument('--save_best_last_only', default=True, type=boolean_string)
    parser.add_argument('--ckpt_async', default=True, type=boolean_string)
    parser.add_argument('--ckpt_num_shards', default=8, type=int)
    parser.add_argument('--ckpt_shard_by_rank', default=True, type=boolean_string)
    parser.add_argument('--ckpt_max_pending', default=1, type=int)
    parser.add_argument('--ckpt_subdir', default='ckpts', type=str)
    parser.add_argument('--ckpt_meta_wait_timeout_s', default=600.0, type=float)

    parser.add_argument('--use_tb', type=boolean_string, default=True)
    parser.add_argument('--tb_dir', type=str, default='')
    parser.add_argument('--tb_log_freq', type=int, default=20)
    parser.add_argument('--tb_vis_freq', type=int, default=500)
    parser.add_argument('--tb_num_vis', type=int, default=2)

    parser.add_argument('--debug_vis', type=boolean_string, default=False)
    parser.add_argument('--debug_vis_num', type=int, default=64)
    parser.add_argument('--debug_vis_freq', type=int, default=200)
    parser.add_argument('--debug_vis_dir', type=str, default='')

    parser.add_argument('--load_weights_path', type=str, default=None)
    parser.add_argument('--freeze_modules', type=list, default=[])
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_epochs', type=int, default=1)
    parser.add_argument('--freeze_losses', type=list, default=[])

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--backbone_path', default='pretrained_checkpoints/resnet50-19c8e357.pth', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=1, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--bert_model', default='pretrained_checkpoints/bert-base-uncased.tar.gz', type=str)
    parser.add_argument('--bert_token_mode', default='pretrained_checkpoints/bert_base_uncased', type=str)
    parser.add_argument('--bert_output_dim', default=768, type=int)
    parser.add_argument('--bert_output_layers', default=12, type=int)
    parser.add_argument('--max_query_len', default=128, type=int)
    parser.add_argument('--bert_enc_num', default=12, type=int)

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--loss_loc', default='loss_boxes', type=str)
    parser.add_argument('--box_xyxy', action='store_true')
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    parser.add_argument("--data_root", default="/mnt/data_2/pl/miccai")
    parser.add_argument("--split_root", default="/mnt/data_2/pl/miccai/best_pathvgsam3/split/data")
    parser.add_argument('--dataset', default='pathology2')
    parser.add_argument('--test_split', default='testB')
    parser.add_argument('--img_size', default=768)
    parser.add_argument('--cache_images', action='store_true')

    parser.add_argument('--output_dir', default='work_dirs/')
    parser.add_argument('--save_pred_path', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--collate_fn', default='collate_fn')

    parser.add_argument('--batch_size_val', default=32, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    parser.add_argument('--train_transforms', default=train_transforms)
    parser.add_argument('--test_transforms', default=test_transforms)

    parser.add_argument('--enable_batch_accum', action='store_true')

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')

    parser.add_argument('--config', default='configs/TransCP_R50_pathology2.py', type=str)
    parser.add_argument('--model_config')

    parser.add_argument('--use_sam3', type=boolean_string, default=True)
    parser.add_argument('--sam3_ckpt', type=str, default='')
    parser.add_argument('--sam3_bpe_path', type=str, default='')
    parser.add_argument('--sam3_resolution', type=int, default=1024)
    parser.add_argument('--sam3_confidence_threshold', type=float, default=0.0)
    parser.add_argument('--sam3_prompt_coord', type=str, default='norm', choices=['norm', 'pixel'])
    parser.add_argument("--debug_sam3_every", default=0, type=int)

    # ✅ best metric settings
    parser.add_argument('--best_thr_testA', default=0.7, type=float)  # testA: Acc@0.70
    parser.add_argument('--best_thr_testB', default=0.5, type=float)  # testB: Acc@0.50

    # ✅ joint best = wA*scoreA + wB*scoreB
    parser.add_argument('--best_joint_wA', default=0.5, type=float)
    parser.add_argument('--best_joint_wB', default=0.5, type=float)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    if getattr(args, "distributed", False) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ install stdout/stderr tee (captures progress prints into iter.txt)
    tee_file = None
    old_out = None
    old_err = None
    try:
        tee_file, old_out, old_err = _install_iter_tee(output_dir, utils.get_rank())
    except Exception:
        tee_file = None
        old_out = None
        old_err = None

    if isinstance(args.debug_vis_dir, str) and args.debug_vis_dir.strip() == "":
        args.debug_vis_dir = str(output_dir / "debug_vis_train")

    logger = get_logger("train", args.output_dir, utils.get_rank(), filename='iter.log')
    epoch_logger = get_logger("train_epoch", args.output_dir, utils.get_rank(), filename='epoch.log')
    logger.info(args)

    device = _resolve_device(args)

    ckpt_sync_group = None
    if getattr(args, "distributed", False) and bool(getattr(args, "ckpt_sync", False)):
        ckpt_sync_group = build_ckpt_sync_group(logger, hours=int(args.ckpt_sync_timeout_hours))
        logger.info(f"[CKPT] ckpt_sync_group={ckpt_sync_group}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    writer = None

    model, criterion, postprocessor = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # =========================
    # ✅ optimizer：不漏参数 + 打印
    # =========================
    param_groups = build_param_groups(
        model_without_ddp,
        args,
        logger=logger if utils.is_main_process() else None
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.lr_drop], gamma=0.1
    )

    dataset_train = build_dataset(test=False, args=args)

    # =========================
    # ✅ build val loaders for testA / testB
    # =========================
    def _build_val_loader_for_split(split_name: str):
        old_split = getattr(args, "test_split", "testB")
        args.test_split = split_name
        ds = build_dataset(test=True, args=args)
        args.test_split = old_split

        if args.distributed:
            sp = DistributedSampler(ds, shuffle=False)
        else:
            sp = torch.utils.data.SequentialSampler(ds)

        dl = DataLoader(
            ds,
            args.batch_size_val,
            sampler=sp,
            pin_memory=args.pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        return ds, dl, sp

    dataset_val_A, data_loader_val_A, sampler_val_A = _build_val_loader_for_split("testA")
    dataset_val_B, data_loader_val_B, sampler_val_B = _build_val_loader_for_split("testB")

    logger.info(f'The size of dataset: train({len(dataset_train)}) testA({len(dataset_val_A)}) testB({len(dataset_val_B)})')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # cache val images (optional)
    for ds, sp in [(dataset_val_A, sampler_val_A), (dataset_val_B, sampler_val_B)]:
        if getattr(ds, "cache_images", False) is True:
            for i in sp:
                ds.cache(i)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    iters_per_epoch = len(data_loader_train)

    # =========================
    # ✅ best trackers
    # =========================
    best_testA = 0.0
    best_testB = 0.0
    best_joint = 0.0

    if args.resume:
        best_testA, best_testB, best_joint = _load_ckpt(
            args.resume, model_without_ddp, optimizer, lr_scheduler, args, logger, device=device
        )
    elif args.load_weights_path:
        # 你的模型里如果实现了这个函数就会用；没有也不影响（你自己知道）
        try:
            model_without_ddp.load_pretrained_weights(args.load_weights_path)
        except Exception as e:
            logger.info(f"[Init] load_pretrained_weights failed: {repr(e)}")

    run_tag = "sam3" if args.use_sam3 else "nosam3"

    if args.use_tb and SummaryWriter is not None and utils.is_main_process():
        tb_dir = args.tb_dir.strip() if isinstance(args.tb_dir, str) else ""
        if tb_dir == "":
            tb_dir = str(output_dir / "tb")
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        purge_step = max(0, int(args.start_epoch) * int(iters_per_epoch))
        writer = SummaryWriter(log_dir=tb_dir, purge_step=purge_step)
        logger.info(f"[TB] logdir={tb_dir}, purge_step={purge_step}")

    ckpt_root = Path(args.output_dir) / str(getattr(args, "ckpt_subdir", "ckpts"))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ckpt_saver = None
    if bool(getattr(args, "ckpt_async", True)):
        ckpt_saver = AsyncShardedCheckpointer(
            logger=logger if utils.is_main_process() else None,
            fsync_dir=bool(getattr(args, "ckpt_fsync_dir", False)),
            max_pending=int(getattr(args, "ckpt_max_pending", 1)),
        )

    # =========================
    # ✅ eval-only: evaluate both splits
    # =========================
    if args.eval:
        with torch.no_grad():
            val_stats_A, val_acc_A, val_time_A = evaluate(
                model, criterion, postprocessor,
                data_loader_val_A, device,
                args.save_pred_path,
                args=args, writer=writer,
                global_step=max(0, int(args.start_epoch) * int(iters_per_epoch)),
                tb_num_vis=int(args.tb_num_vis),
                tb_tag_prefix=run_tag + "/testA"
            )
            score_A = _pick_acc_at(val_acc_A, float(getattr(args, "best_thr_testA", 0.7)))

            val_stats_B, val_acc_B, val_time_B = evaluate(
                model, criterion, postprocessor,
                data_loader_val_B, device,
                args.save_pred_path,
                args=args, writer=writer,
                global_step=max(0, int(args.start_epoch) * int(iters_per_epoch)),
                tb_num_vis=int(args.tb_num_vis),
                tb_tag_prefix=run_tag + "/testB"
            )
            score_B = _pick_acc_at(val_acc_B, float(getattr(args, "best_thr_testB", 0.5)))

            wA = float(getattr(args, "best_joint_wA", 0.5))
            wB = float(getattr(args, "best_joint_wB", 0.5))
            if abs(wA) + abs(wB) < 1e-12:
                wA, wB = 0.5, 0.5
            joint_score = wA * score_A + wB * score_B

        logger.info(' '.join(['[Eval testA]', *[f'{k}: {v:.4f}' for k, v in val_stats_A.items()]]))
        logger.info(' '.join(['[Acc  testA]', *[f'{k}: {v:.4f}' for k, v in val_acc_A.items()]]))
        logger.info(f"[Score testA] Acc@{float(args.best_thr_testA):.2f} = {score_A:.6f}")

        logger.info(' '.join(['[Eval testB]', *[f'{k}: {v:.4f}' for k, v in val_stats_B.items()]]))
        logger.info(' '.join(['[Acc  testB]', *[f'{k}: {v:.4f}' for k, v in val_acc_B.items()]]))
        logger.info(f"[Score testB] Acc@{float(args.best_thr_testB):.2f} = {score_B:.6f}")

        logger.info(f"[Score joint] {wA:.2f}*A + {wB:.2f}*B = {joint_score:.6f}")

        if writer is not None:
            writer.flush()

        # restore tee
        try:
            if old_out is not None:
                sys.stdout = old_out
            if old_err is not None:
                sys.stderr = old_err
            if tee_file is not None:
                tee_file.flush()
                tee_file.close()
        except Exception:
            pass
        return

    epoch_trainer = train_one_epoch_w_accum if args.enable_batch_accum else train_one_epoch
    logger.info("Start training")
    start_time = time.time()

    filter_keywords = _parse_keywords(args.ckpt_filter_keywords) if bool(args.ckpt_filter_enable) else []

    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        global_step_start = epoch * iters_per_epoch

        train_stats = epoch_trainer(
            model, criterion,
            data_loader_train, optimizer,
            device, epoch, args.epochs,
            args.clip_max_norm,
            args=args,
            writer=writer,
            global_step_start=int(global_step_start),
            tb_log_freq=int(args.tb_log_freq),
            tb_vis_freq=int(args.tb_vis_freq),
            tb_num_vis=int(args.tb_num_vis),
            postprocessor=postprocessor,
            tb_tag_prefix=run_tag,
            debug_vis=bool(args.debug_vis),
            debug_vis_freq=int(args.debug_vis_freq),
            debug_vis_num=int(args.debug_vis_num),
            debug_vis_dir=str(args.debug_vis_dir),
        )

        lr_scheduler.step()

        # =========================
        # ✅ evaluate testA / testB
        # =========================
        val_stats_A, val_acc_A, _ = evaluate(
            model, criterion, postprocessor,
            data_loader_val_A, device,
            args.save_pred_path,
            args=args,
            writer=writer,
            global_step=int((epoch + 1) * iters_per_epoch),
            tb_num_vis=int(args.tb_num_vis),
            tb_tag_prefix=run_tag + "/testA"
        )
        score_A = _pick_acc_at(val_acc_A, float(getattr(args, "best_thr_testA", 0.7)))

        val_stats_B, val_acc_B, _ = evaluate(
            model, criterion, postprocessor,
            data_loader_val_B, device,
            args.save_pred_path,
            args=args,
            writer=writer,
            global_step=int((epoch + 1) * iters_per_epoch),
            tb_num_vis=int(args.tb_num_vis),
            tb_tag_prefix=run_tag + "/testB"
        )
        score_B = _pick_acc_at(val_acc_B, float(getattr(args, "best_thr_testB", 0.5)))

        wA = float(getattr(args, "best_joint_wA", 0.5))
        wB = float(getattr(args, "best_joint_wB", 0.5))
        if abs(wA) + abs(wB) < 1e-12:
            wA, wB = 0.5, 0.5
        joint_score = wA * score_A + wB * score_B

        improved_A = False
        improved_B = False
        improved_J = False

        if utils.is_main_process():
            improved_A = (score_A > best_testA + 1e-12)
            improved_B = (score_B > best_testB + 1e-12)
            improved_J = (joint_score > best_joint + 1e-12)
            if improved_A:
                best_testA = score_A
            if improved_B:
                best_testB = score_B
            if improved_J:
                best_joint = joint_score

        if args.distributed:
            # ✅ float32那个：best 同步用 float32（原来是 float64）
            t = torch.tensor(
                [1.0 if improved_A else 0.0, float(best_testA),
                 1.0 if improved_B else 0.0, float(best_testB),
                 1.0 if improved_J else 0.0, float(best_joint)],
                device=device, dtype=torch.float32
            )
            dist.broadcast(t, src=0)
            improved_A = bool(int(t[0].item()))
            best_testA = float(t[1].item())
            improved_B = bool(int(t[2].item()))
            best_testB = float(t[3].item())
            improved_J = bool(int(t[4].item()))
            best_joint = float(t[5].item())

        save_best_last_only = bool(getattr(args, "save_best_last_only", True))
        want_save_best_A = improved_A
        want_save_best_B = improved_B
        want_save_best_J = improved_J
        want_save_last = ((epoch + 1) == int(args.epochs))

        # ✅ 新增：最后 K 个 epoch 都保存（比如 epochs=120, K=3 => 118/119/120）
        last_k = max(1, int(getattr(args, "save_last_k", 3)))
        tail_start = max(1, int(args.epochs) - last_k + 1)
        want_save_tail_k = ((epoch + 1) >= tail_start)

        want_save_latest = False
        want_save_periodic = False
        if not save_best_last_only:
            want_save_latest = True
            want_save_periodic = (int(args.checkpoint_step) > 0 and ((epoch + 1) % int(args.checkpoint_step) == 0))

        if (want_save_best_A or want_save_best_B or want_save_best_J or want_save_last or want_save_tail_k or want_save_latest or want_save_periodic):

            if args.output_dir and getattr(args, "distributed", False) and bool(getattr(args, "ckpt_sync", False)):
                ckpt_barrier(ckpt_sync_group, logger=logger, tag="pre-save")

            model_sd = model_without_ddp.state_dict()
            if filter_keywords:
                model_sd = filter_state_dict(model_sd, filter_keywords)

            shard_by_rank = bool(getattr(args, "ckpt_shard_by_rank", True)) and bool(args.distributed)
            world_size = utils.get_world_size()
            rank = utils.get_rank()

            # ✅ 是否强制保存 float32（你说的 float32 那个）
            force_fp32_snapshot = bool(getattr(args, "ckpt_force_cpu_snapshot", False))

            def _enqueue_save(tag: str):
                ckpt_dir = ckpt_root / tag
                _ensure_dir(ckpt_dir)

                metricA = f"Acc@{float(getattr(args, 'best_thr_testA', 0.7)):.2f}"
                metricB = f"Acc@{float(getattr(args, 'best_thr_testB', 0.5)):.2f}"
                metricJ = f"{wA:.2f}*{metricA}+{wB:.2f}*{metricB}"

                if shard_by_rank:
                    shard_files = [f"rank{r:03d}.pth" for r in range(world_size)]
                    my_shard_file = f"rank{rank:03d}.pth"

                    keys = sorted(list(model_sd.keys()))
                    my_keys = keys[rank::world_size]

                    # ✅ 保存前：搬到 CPU，可选强制 float32
                    my_sd_cpu = _to_cpu_state_dict({k: model_sd[k] for k in my_keys}, force_fp32=force_fp32_snapshot)

                    meta_payload = None
                    if utils.is_main_process():
                        meta_payload = {
                            "epoch": int(epoch),

                            # ✅ best metrics
                            "best_testA": float(best_testA),
                            "best_testB": float(best_testB),
                            "best_joint": float(best_joint),
                            "best_metric_testA": metricA,
                            "best_metric_testB": metricB,
                            "best_metric_joint": metricJ,

                            # ✅ latest scores for debugging
                            "val_acc_testA": val_acc_A,
                            "val_acc_testB": val_acc_B,
                            "val_score_testA": float(score_A),
                            "val_score_testB": float(score_B),
                            "val_joint_score": float(joint_score),

                            "args": sanitize_args_for_ckpt(args),
                            "rng_state": _get_rng_state(),
                            "shards": shard_files,
                            "shard_by_rank": True,
                            "world_size": int(world_size),
                            "meta_wait_timeout_s": float(getattr(args, "ckpt_meta_wait_timeout_s", 600.0)),
                            "tag": str(tag),
                            "uuid": str(uuid.uuid4()),

                            # ✅ 记录是否强制 fp32
                            "ckpt_force_cpu_snapshot": bool(force_fp32_snapshot),
                        }

                    job = dict(
                        ckpt_dir=ckpt_dir,
                        tag=tag,
                        model_sd_cpu=my_sd_cpu,
                        meta_payload=meta_payload,
                        shard_filenames=[my_shard_file],
                        logger=logger if utils.is_main_process() else None,
                        fsync_dir=bool(getattr(args, "ckpt_fsync_dir", False)),
                    )

                    if ckpt_saver is not None:
                        ckpt_saver.submit(job)
                    else:
                        _save_sharded_checkpoint_dir(**job)
                    return

                # non-distributed sharding (rank0 only)
                if not utils.is_main_process():
                    return

                keys = sorted(list(model_sd.keys()))
                num_shards = max(1, int(getattr(args, "ckpt_num_shards", 8)))
                key_shards = _split_keys_even(keys, num_shards)
                shard_files = [f"shard{i:03d}.pth" for i in range(len(key_shards))]

                meta_payload = {
                    "epoch": int(epoch),

                    "best_testA": float(best_testA),
                    "best_testB": float(best_testB),
                    "best_joint": float(best_joint),
                    "best_metric_testA": metricA,
                    "best_metric_testB": metricB,
                    "best_metric_joint": metricJ,

                    "val_acc_testA": val_acc_A,
                    "val_acc_testB": val_acc_B,
                    "val_score_testA": float(score_A),
                    "val_score_testB": float(score_B),
                    "val_joint_score": float(joint_score),

                    "args": sanitize_args_for_ckpt(args),
                    "rng_state": _get_rng_state(),
                    "shards": shard_files,
                    "shard_by_rank": False,
                    "world_size": 1,
                    "meta_wait_timeout_s": float(getattr(args, "ckpt_meta_wait_timeout_s", 600.0)),
                    "tag": str(tag),
                    "uuid": str(uuid.uuid4()),

                    # ✅ 记录是否强制 fp32
                    "ckpt_force_cpu_snapshot": bool(force_fp32_snapshot),
                }

                for i, ks in enumerate(key_shards):
                    # ✅ 保存前：搬到 CPU，可选强制 float32
                    sd_cpu = _to_cpu_state_dict({k: model_sd[k] for k in ks}, force_fp32=force_fp32_snapshot)
                    job = dict(
                        ckpt_dir=ckpt_dir,
                        tag=tag,
                        model_sd_cpu=sd_cpu,
                        meta_payload=(meta_payload if i == len(key_shards) - 1 else None),
                        shard_filenames=[shard_files[i]],
                        logger=logger,
                        fsync_dir=bool(getattr(args, "ckpt_fsync_dir", False)),
                    )
                    if ckpt_saver is not None:
                        ckpt_saver.submit(job)
                    else:
                        _save_sharded_checkpoint_dir(**job)

            # ✅ 统一收集要保存的 tag，并去重（避免 tail_k 与 periodic 同时触发重复写 epoch_XXXX）
            tags_to_save: List[str] = []
            if want_save_best_A:
                tags_to_save.append("best_testA")
            if want_save_best_B:
                tags_to_save.append("best_testB")
            if want_save_best_J:
                tags_to_save.append("best_joint")

            if want_save_tail_k:
                tags_to_save.append(f"epoch_{epoch+1:04d}")

            if want_save_last:
                tags_to_save.append("last")

            if (not save_best_last_only) and want_save_latest:
                tags_to_save.append("latest")
            if (not save_best_last_only) and want_save_periodic:
                tags_to_save.append(f"epoch_{epoch+1:04d}")

            # 去重但保序
            _seen_tags = set()
            uniq_tags = []
            for ttag in tags_to_save:
                if ttag in _seen_tags:
                    continue
                _seen_tags.add(ttag)
                uniq_tags.append(ttag)

            for ttag in uniq_tags:
                _enqueue_save(ttag)

            if args.output_dir and getattr(args, "distributed", False) and bool(getattr(args, "ckpt_sync", False)):
                ckpt_barrier(ckpt_sync_group, logger=logger, tag="post-save")

        if args.output_dir and utils.is_main_process():
            epoch_logger.info(' '.join([f'Epoch [{epoch + 1}](train stats)', *[f'train_{k}: {v:.4f}' for k, v in train_stats.items()]]))

            epoch_logger.info(' '.join([f'Epoch [{epoch + 1}](testA stats)', *[f'testA_{k}: {v:.4f}' for k, v in val_stats_A.items()]]))
            epoch_logger.info(' '.join([f'Epoch [{epoch + 1}](testA acc)', *[f'{k}: {v:.4f}' for k, v in val_acc_A.items()]]))
            epoch_logger.info(f"[Best@testA] metric=Acc@{float(args.best_thr_testA):.2f}  best={best_testA:.6f}  cur={score_A:.6f}")

            epoch_logger.info(' '.join([f'Epoch [{epoch + 1}](testB stats)', *[f'testB_{k}: {v:.4f}' for k, v in val_stats_B.items()]]))
            epoch_logger.info(' '.join([f'Epoch [{epoch + 1}](testB acc)', *[f'{k}: {v:.4f}' for k, v in val_acc_B.items()]]))
            epoch_logger.info(f"[Best@testB] metric=Acc@{float(args.best_thr_testB):.2f}  best={best_testB:.6f}  cur={score_B:.6f}")

            epoch_logger.info(f"[Best@joint] metric={wA:.2f}*Acc@{float(args.best_thr_testA):.2f}+{wB:.2f}*Acc@{float(args.best_thr_testB):.2f}  best={best_joint:.6f}  cur={joint_score:.6f}")
            epoch_logger.info('')

    total_time = time.time() - start_time
    logger.info('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass

    if ckpt_saver is not None:
        try:
            ckpt_saver.close(wait=True)
        except Exception:
            pass

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # ✅ restore stdout/stderr and close tee file
    try:
        if old_out is not None:
            sys.stdout = old_out
        if old_err is not None:
            sys.stderr = old_err
        if tee_file is not None:
            tee_file.flush()
            tee_file.close()
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransCP training script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
