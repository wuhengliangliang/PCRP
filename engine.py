# engine.py
# -*- coding: utf-8 -*-

import os
import math
import time
import datetime
from typing import Iterable, Dict, Any, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

import util.misc as utils
from util import box_ops


# -------------------------
# CUDA Prefetcher (optional)
# -------------------------
class data_prefetcher:
    """
    可选加速模块：通过 CUDA stream 让 “Host->Device 拷贝” 与 “GPU 计算” 重叠，从而提高吞吐。

    适用前提：
      - device.type == "cuda"
      - dataloader 每次 yield (samples, targets)
      - samples 是 NestedTensor，支持 samples.decompose() -> (img, mask)
      - targets 是 dict 或者一个对象，支持 targets.tensor_dict

    思路：
      - 在一个独立 CUDA stream 里预先把下一批数据 copy 到 GPU
      - 当前 batch 计算时，下一批 copy 并行进行
      - next() 时用 current_stream().wait_stream() 保证数据 copy 已完成
    """
    def __init__(self, loader, device: torch.device):
        self.length = len(loader)               # dataloader 总长度（batch 数）
        self.loader = iter(loader)              # 变成 iterator
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.device = device
        self.next_samples = None                # 预取的下一批图像/mask
        self.next_targets = None                # 预取的下一批 targets dict
        self.preload()                          # 初始化时先预取一批

    def preload(self):
        """
        把 dataloader 的下一批拿出来，并（若 cuda）异步搬到 GPU。
        """
        try:
            samples, targets = next(self.loader)
        except StopIteration:
            # dataloader 结束
            self.next_samples = None
            self.next_targets = None
            return

        # CPU 情况：不需要 stream 预取，直接保存
        if self.stream is None:
            self.next_samples = samples
            self.next_targets = targets
            return

        # CUDA 情况：在独立 stream 中做 H2D copy
        with torch.cuda.stream(self.stream):
            img, mask = samples.decompose()
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            self.next_samples = (img, mask)

            # targets 兼容 dict 或对象.tensor_dict
            tdict = targets if isinstance(targets, dict) else targets.tensor_dict
            moved = {}
            for k, v in tdict.items():
                if torch.is_tensor(v):
                    moved[k] = v.to(self.device, non_blocking=True)
                else:
                    moved[k] = v
            self.next_targets = moved

    def next(self):
        """
        返回当前已经预取好的 batch，并立即启动下一次 preload()。
        """
        if self.stream is not None:
            # 当前 stream 等待预取 stream，保证数据 copy 完成
            torch.cuda.current_stream().wait_stream(self.stream)

        samples = self.next_samples
        targets = self.next_targets
        if samples is None:
            return None, None

        # 立刻预取下一批
        self.preload()
        return samples, targets


# -------------------------
# Small helpers
# -------------------------
def _is_dist() -> bool:
    """
    判断 DDP 是否初始化。
    优先用 utils.is_dist_avail_and_initialized（如果你的 util.misc 提供），否则用 torch.distributed 的原生判断。
    """
    return utils.is_dist_avail_and_initialized() if hasattr(utils, "is_dist_avail_and_initialized") else (
        dist.is_available() and dist.is_initialized()
    )


def _rank() -> int:
    """当前进程 rank；非分布式默认 0。"""
    return dist.get_rank() if _is_dist() else 0


def _is_rank0() -> bool:
    """只有 rank0 执行可视化/保存/打印，避免多卡重复写文件。"""
    return _rank() == 0


def _safe_get_targets_dict(targets) -> Dict[str, Any]:
    """
    兼容两种 targets：
      - dict
      - 带 tensor_dict 属性的对象（你数据集里常见）
    """
    return targets if isinstance(targets, dict) else targets.tensor_dict


def _to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    """
    把一个 (3,H,W) 的 float 图像转成 uint8：
      - 先取 min/max 做线性归一化到 [0,1]
      - 再乘 255
    注意：这是为了 debug 可视化，不影响训练。
    """
    x = x.detach().float().cpu()
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < 1e-6:
        x = torch.zeros_like(x)
    else:
        x = (x - xmin) / (xmax - xmin)
    x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    return x


def _cxcywh_norm_to_xyxy_pix(box_cxcywh: torch.Tensor, H: int, W: int) -> Tuple[float, float, float, float]:
    """
    把 normalized 的 cxcywh（[0,1]）转成像素 xyxy：
      cx,cy,w,h 都是相对比例
      x1=(cx-0.5*w)*W, y1=(cy-0.5*h)*H
      x2=(cx+0.5*w)*W, y2=(cy+0.5*h)*H
    """
    cx, cy, bw, bh = box_cxcywh.unbind(-1)
    x1 = (cx - 0.5 * bw) * W
    y1 = (cy - 0.5 * bh) * H
    x2 = (cx + 0.5 * bw) * W
    y2 = (cy + 0.5 * bh) * H
    return float(x1), float(y1), float(x2), float(y2)


def _box_iou_cxcywh_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    计算两框 IoU（输入是 normalized cxcywh）。
    这里手写 IoU 是为了 debug 文本显示用，避免依赖更多函数。
    """
    ax1 = a[0] - 0.5 * a[2]
    ay1 = a[1] - 0.5 * a[3]
    ax2 = a[0] + 0.5 * a[2]
    ay2 = a[1] + 0.5 * a[3]

    bx1 = b[0] - 0.5 * b[2]
    by1 = b[1] - 0.5 * b[3]
    bx2 = b[0] + 0.5 * b[2]
    by2 = b[1] + 0.5 * b[3]

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)

    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = torch.clamp(area_a + area_b - inter, min=1e-6)
    return float((inter / union).item())


def _draw_boxes_and_mask(
    img_uint8_chw: torch.Tensor,
    b0: Optional[torch.Tensor],
    b1: Optional[torch.Tensor],
    gt: Optional[torch.Tensor],
    sam_mask: Optional[torch.Tensor],
    save_path: str,
    extra_text: Optional[str] = None,
):
    """
    保存一张 debug 图：
      - 原图（img_uint8_chw）转 PIL
      - 可选 overlay SAM mask（>0.5 的区域用半透明红色覆盖）
      - 画 GT / B0 / B1 三个框
      - 写额外文字（IoU、sam_score 等）
    """
    img = img_uint8_chw.permute(1, 2, 0).contiguous().numpy()
    pil = Image.fromarray(img)
    W, H = pil.size

    # 1) 叠加 SAM mask（这里用 Image.composite 做硬阈值遮罩）
    if sam_mask is not None:
        m = sam_mask.detach().float().cpu()
        if m.dim() == 3:
            m = m[0]
        m = (m > 0.5).to(torch.uint8) * 140          # 140 是遮罩强度（灰度）
        m_pil = Image.fromarray(m.numpy(), mode="L")
        red = Image.new("RGB", (W, H), (255, 0, 0))  # 红色 overlay
        pil = Image.composite(red, pil, m_pil)

    draw = ImageDraw.Draw(pil)

    def _draw_one(box, color, tag):
        """画一个框并写 tag。box 输入是 normalized cxcywh。"""
        if box is None:
            return
        x1, y1, x2, y2 = _cxcywh_norm_to_xyxy_pix(box, H, W)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), tag, fill=color)

    # 约定颜色：GT 绿，B0 黄，B1 青
    _draw_one(gt, (0, 255, 0), "GT")
    _draw_one(b0, (255, 255, 0), "B0")
    _draw_one(b1, (0, 255, 255), "B1")

    if extra_text:
        draw.text((6, 6), extra_text, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil.save(save_path)


def _get_amp_enabled(args) -> bool:
    """
    从 args 判断是否开启 AMP：
      - args.amp / args.use_amp / args.fp16 任意一个 True 就开启
    """
    if args is None:
        return False
    return bool(getattr(args, "amp", False) or getattr(args, "use_amp", False) or getattr(args, "fp16", False))


def _get_accum_iter(args) -> int:
    """
    从 args 判断梯度累积步数：
      - args.enable_batch_accum=True 时返回 args.accum_iter（默认 2）
      - 否则返回 1
    """
    if args is None:
        return 1
    if bool(getattr(args, "enable_batch_accum", False)):
        return int(getattr(args, "accum_iter", 2))
    return 1


def _autocast_ctx(use_amp: bool, device: torch.device):
    """
    返回 autocast 上下文：
      - CUDA + use_amp=True -> 开启 autocast
      - 否则 -> nullcontext（什么都不做）

    兼容新旧写法：
      - 新版推荐 torch.amp.autocast(device_type="cuda")
      - 老版本 fallback torch.cuda.amp.autocast
    """
    if not (use_amp and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    max_norm: float = 0.0,
    args=None,
    writer=None,
    global_step_start: int = 0,
    tb_log_freq: int = 20,
    debug_vis: bool = False,
    debug_vis_freq: int = 200,
    debug_vis_num: int = 8,
    debug_vis_dir: str = "outputs/debug_vis_train",
    **kwargs,
):
    """
    训练一个 epoch。

    ✅ 特性：
      - 支持 AMP（GradScaler）
      - 支持 梯度累积（accum_iter）
      - 支持 prefetcher（可选）
      - 兼容 train.py 的调用签名（epoch/epochs/args 等）
      - 可选 debug 可视化（保存 b0/b1/gt 以及 SAM mask overlay 图）

    参数说明（只提关键的）：
      epoch/epochs: 当前 epoch 和总 epoch（用于日志、可视化文件名）
      max_norm: 梯度裁剪阈值（<=0 表示不裁剪）
      writer: TensorBoard writer（rank0 才写）
      global_step_start: 外部传入的 step 起始（用于跨 epoch 连续记录）
      debug_vis_*: 控制可视化频率和保存目录
    """
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f"Epoch: [{epoch}/{epochs}]"

    # AMP 与梯度累积配置
    use_amp = _get_amp_enabled(args)
    accum_iter = _get_accum_iter(args)

    # GradScaler：只有 cuda + use_amp 才启用
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    # 是否启用 prefetcher
    use_prefetch = bool(getattr(args, "use_prefetcher", False)) if args is not None else False

    # -------------------------
    # 1) Prefetcher 路径：自己控制循环
    # -------------------------
    if use_prefetch and device.type == "cuda":
        prefetcher = data_prefetcher(data_loader, device)
        data = prefetcher.next()
        step = 0
        while data[0] is not None:
            (img, mask), tdict = data
            data = prefetcher.next()

            # 核心训练逻辑封装到 _train_step
            _train_step(
                model, criterion, optimizer, scaler,
                img, mask, tdict, device,
                epoch=epoch, step=step,
                global_step_start=global_step_start,
                max_norm=max_norm,
                use_amp=use_amp,
                accum_iter=accum_iter,
                metric_logger=metric_logger,
                writer=writer,
                tb_log_freq=tb_log_freq,
                debug_vis=debug_vis,
                debug_vis_freq=debug_vis_freq,
                debug_vis_num=debug_vis_num,
                debug_vis_dir=debug_vis_dir,
            )
            step += 1

        metric_logger.synchronize_between_processes()
        return {k: m.global_avg for k, m in metric_logger.meters.items()}

    # -------------------------
    # 2) 普通 dataloader 路径：用 metric_logger.log_every 打印进度
    # -------------------------
    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # NestedTensor -> (img,mask)
        img, mask = samples.decompose()
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # targets -> dict，并把 tensor 全部搬到 device
        tdict = _safe_get_targets_dict(targets)
        tdict = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tdict.items()}

        _train_step(
            model, criterion, optimizer, scaler,
            img, mask, tdict, device,
            epoch=epoch, step=step,
            global_step_start=global_step_start,
            max_norm=max_norm,
            use_amp=use_amp,
            accum_iter=accum_iter,
            metric_logger=metric_logger,
            writer=writer,
            tb_log_freq=tb_log_freq,
            debug_vis=debug_vis,
            debug_vis_freq=debug_vis_freq,
            debug_vis_num=debug_vis_num,
            debug_vis_dir=debug_vis_dir,
        )

    metric_logger.synchronize_between_processes()
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


def _train_step(
    model, criterion, optimizer, scaler,
    img, mask, tdict, device,
    epoch: int, step: int,
    global_step_start: int,
    max_norm: float,
    use_amp: bool,
    accum_iter: int,
    metric_logger: utils.MetricLogger,
    writer=None,
    tb_log_freq: int = 20,
    debug_vis: bool = False,
    debug_vis_freq: int = 200,
    debug_vis_num: int = 8,
    debug_vis_dir: str = "outputs/debug_vis_train",
):
    """
    单步训练（一个 batch）。

    你这里做了几件关键事：
      1) 从 tdict 取文本、reason、GT bbox
      2) forward 调 model（传入 gt_bbox/epoch/step，用于 SAM3 prompt curriculum + debug）
      3) criterion 计算 loss_dict，并按 weight_dict 加权求和
      4) 支持 AMP + 梯度累积（accum_iter）
      5) rank0 做 TensorBoard 记录
      6) rank0 可选保存 debug 图片（b0/b1/gt + sam mask overlay）
    """
    # ---------- 取主表达 ----------
    word_id = tdict["word_id"]
    word_mask = tdict["word_mask"]

    # ---------- 取 reasoning tokens（可选） ----------
    r0 = tdict.get("reason_id_0", None)
    rm0 = tdict.get("reason_mask_0", None)
    r1 = tdict.get("reason_id_1", None)
    rm1 = tdict.get("reason_mask_1", None)
    r2 = tdict.get("reason_id_2", None)
    rm2 = tdict.get("reason_mask_2", None)

    # 数据集中如果没有 reason，就构造一个 dummy（长度 1）
    # 注意：rm0 用 1 表示 mask（ignored），等价于“空输入”
    if r0 is None:
        B = word_id.shape[0]
        r0 = torch.zeros((B, 1), device=device, dtype=word_id.dtype)
        rm_dtype = word_mask.dtype if torch.is_tensor(word_mask) else torch.bool
        rm0 = torch.ones((B, 1), device=device, dtype=rm_dtype)
        r1, rm1, r2, rm2 = r0.clone(), rm0.clone(), r0.clone(), rm0.clone()

    # GT bbox：训练时用于 loss，也用于 SAM3 warmup/mix prompt
    gt_bbox = tdict.get("bbox", None)  # (B,4) cxcywh norm

    global_step = int(global_step_start) + int(step)

    # ---------- forward + loss（可 AMP） ----------
    with _autocast_ctx(use_amp, device):
        outputs = model(
            img, mask,
            word_id, word_mask,
            r0, rm0, r1, rm1, r2, rm2,
            gt_bbox=gt_bbox,
            epoch=int(epoch),
            step=int(global_step),
        )
        loss_dict = criterion(outputs, tdict)
        weight_dict = criterion.weight_dict
        # 按 weight_dict 求总 loss
        losses = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys() if k in weight_dict)

    # reduce_dict：把每张卡的 loss 做 all-reduce（用于日志）
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    loss_dict_reduced_unscaled = {k: float(v.item()) for k, v in loss_dict_reduced.items()}

    # losses_reduced：用于日志显示的总 loss（没有除 accum_iter）
    losses_reduced = sum(loss_dict_reduced[k] * weight_dict.get(k, 1.0)
                         for k in loss_dict_reduced.keys() if k in weight_dict)

    # ---------- 梯度累积：每 accum_iter 步才真正 optimizer.step ----------
    losses = losses / max(1, accum_iter)

    if scaler.is_enabled():
        scaler.scale(losses).backward()
    else:
        losses.backward()

    if (step + 1) % max(1, accum_iter) == 0:
        # 梯度裁剪（注意：AMP 要先 unscale_）
        if max_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer step
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # ---------- 记录 metric ----------
    metric_logger.update(loss=float(losses_reduced.item()))
    for k, v in loss_dict_reduced_unscaled.items():
        metric_logger.update(**{k: v})
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # ---------- TensorBoard（rank0 才写） ----------
    if writer is not None and _is_rank0():
        if tb_log_freq > 0 and (global_step % int(tb_log_freq) == 0):
            writer.add_scalar("train/loss", float(losses_reduced.item()), global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                writer.add_scalar(f"train/{k}", float(v), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

    # ---------- debug 可视化：保存叠加图 ----------
    if debug_vis and _is_rank0() and (int(step) % int(debug_vis_freq) == 0):
        B = img.shape[0]
        num = min(B, int(debug_vis_num))

        # b1：最终输出框（stage-2），通常 shape (B,1,4)
        b1_all = outputs.get("pred_boxes", None)
        if b1_all is not None and b1_all.dim() == 3:
            b1_all = b1_all[:, 0]

        # b0：stage-1 输出框，从 aux_outputs[0] 里取
        b0_all = None
        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], (list, tuple)) and len(outputs["aux_outputs"]) > 0:
            aux0 = outputs["aux_outputs"][0]
            if isinstance(aux0, dict) and "pred_boxes" in aux0:
                b0_all = aux0["pred_boxes"]
                if b0_all is not None and b0_all.dim() == 3:
                    b0_all = b0_all[:, 0]

        # SAM3 相关输出（如果 model 返回了）
        sam_mask = outputs.get("sam3_mask", None)
        sam_score = outputs.get("sam3_score", None)
        sam_rel = outputs.get("sam3_reliability", None)

        for b in range(num):
            img_u8 = _to_uint8_image(img[b])

            b0 = b0_all[b].detach().float().cpu() if b0_all is not None else None
            b1 = b1_all[b].detach().float().cpu() if b1_all is not None else None
            gt = gt_bbox[b].detach().float().cpu() if gt_bbox is not None else None

            sam_b = None
            if sam_mask is not None and torch.is_tensor(sam_mask):
                sam_b = sam_mask[b].detach().float().cpu()

            # 额外信息：IoU(b0,gt) / IoU(b1,gt) / IoU(b0,b1) + sam_score/sam_rel
            extra = []
            try:
                if gt is not None and b0 is not None:
                    extra.append(f"IoU(b0,gt)={_box_iou_cxcywh_norm(b0, gt):.3f}")
                if gt is not None and b1 is not None:
                    extra.append(f"IoU(b1,gt)={_box_iou_cxcywh_norm(b1, gt):.3f}")
                if b0 is not None and b1 is not None:
                    extra.append(f"IoU(b0,b1)={_box_iou_cxcywh_norm(b0, b1):.3f}")
            except Exception:
                pass
            try:
                if sam_score is not None and torch.is_tensor(sam_score):
                    extra.append(f"sam_score={float(sam_score[b].item()):.3f}")
                if sam_rel is not None and torch.is_tensor(sam_rel):
                    extra.append(f"sam_rel={float(sam_rel[b].item()):.3f}")
            except Exception:
                pass

            extra_text = "  ".join(extra) if len(extra) > 0 else None

            save_path = os.path.join(
                str(debug_vis_dir),
                f"train_e{epoch:03d}_gs{global_step:07d}_b{b}_rank{_rank()}.png"
            )
            _draw_boxes_and_mask(img_u8, b0, b1, gt, sam_b, save_path, extra_text=extra_text)

    return float(losses_reduced.item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader: Iterable,
    device: torch.device,
    save_pred_path: str = "",
    args=None,
    writer=None,
    global_step: int = 0,
    tb_num_vis: int = 2,
    tb_tag_prefix: str = "",
    **kwargs,
):
    """
    评估函数（无梯度）。

    ✅返回三个值，兼容 train.py：
      val_stats: loss 等统计（来自 metric_logger）
      val_acc  : Acc@0.50 ~ Acc@0.90 + Mean_iou
      val_time : {"sec": total_seconds}

    其中 Acc@t 的计算方式是：
      对每个样本算 IoU
      IoU > t 记为命中
      全集命中数 / 总样本数
    """
    if postprocessors is None and ("postprocessor" in kwargs):
        postprocessors = kwargs["postprocessor"]

    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    use_amp = _get_amp_enabled(args)

    # IoU 阈值从 0.50 到 0.90，步长 0.05，一共 9 个
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0, 9)], device=device)

    # accum_acc: (9,) 累积命中样本数（每个阈值一个）
    accum_acc = torch.zeros((len(iou_thrs),), device=device, dtype=torch.float32)
    # accum_iou: 累积 IoU 和
    accum_iou = torch.zeros((), device=device, dtype=torch.float32)
    # accum_sample: 累积样本数
    accum_sample = torch.zeros((), device=device, dtype=torch.float32)

    save_preds = (isinstance(save_pred_path, str) and save_pred_path.strip() != "")
    all_pred_ious = []
    all_pred_boxes = []

    start = time.time()

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        img, mask = samples.decompose()
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        tdict = _safe_get_targets_dict(targets)
        tdict = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tdict.items()}

        word_id = tdict["word_id"]
        word_mask = tdict["word_mask"]

        # reason tokens 同 train 的处理：没有就 dummy
        r0 = tdict.get("reason_id_0", None)
        rm0 = tdict.get("reason_mask_0", None)
        r1 = tdict.get("reason_id_1", None)
        rm1 = tdict.get("reason_mask_1", None)
        r2 = tdict.get("reason_id_2", None)
        rm2 = tdict.get("reason_mask_2", None)

        if r0 is None:
            B = word_id.shape[0]
            r0 = torch.zeros((B, 1), device=device, dtype=word_id.dtype)
            rm_dtype = word_mask.dtype if torch.is_tensor(word_mask) else torch.bool
            rm0 = torch.ones((B, 1), device=device, dtype=rm_dtype)
            r1, rm1, r2, rm2 = r0.clone(), rm0.clone(), r0.clone(), rm0.clone()

        # forward（eval 时一般不传 gt_bbox）
        with _autocast_ctx(use_amp, device):
            outputs = model(
                img, mask,
                word_id, word_mask,
                r0, rm0, r1, rm1, r2, rm2,
                epoch=0,
                step=int(global_step) + int(step),
            )

            # 如果有 criterion，就算 loss（用于日志）
            if criterion is not None:
                loss_dict = criterion(outputs, tdict)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict.get(k, 1.0)
                             for k in loss_dict.keys() if k in weight_dict)

        # 记录 loss（reduce 后）
        if criterion is not None:
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            metric_logger.update(loss=float(losses.item()))
            for k, v in loss_dict_reduced.items():
                metric_logger.update(**{k: float(v.item())})

        # ---------------------------------------------------------
        # IoU 计算路径 A：如果 targets 提供 orig_bbox（像素 xyxy）且有 postprocessor
        #   pred_boxes = postprocessors(outputs, tdict) -> 像素 xyxy
        #   ious = box_pair_iou(gt_bbox, pred_bbox)
        # ---------------------------------------------------------
        if ("orig_bbox" in tdict) and (postprocessors is not None):
            gt_bbox = tdict["orig_bbox"]                 # (B,4) xyxy pixel
            pred_boxes = postprocessors(outputs, tdict)  # (B,4) xyxy pixel
            ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]  # (B,)
        else:
            # -----------------------------------------------------
            # IoU 计算路径 B：用 normalized cxcywh（不依赖 postprocessor）
            #   gt_cxcywh: tdict["bbox"]
            #   pred: outputs["pred_boxes"] -> (B,1,4) or (B,4)
            #   转 xyxy 再算 iou
            # -----------------------------------------------------
            if "bbox" in tdict and "pred_boxes" in outputs:
                gt_cxcywh = tdict["bbox"]
                pred = outputs["pred_boxes"]
                if pred.dim() == 3:
                    pred = pred[:, 0]
                gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_cxcywh)
                pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred)
                ious = box_ops.box_pair_iou(gt_xyxy, pred_xyxy)[0]
            else:
                continue

        # ---------------------------------------------------------
        # 统计 Acc@thr：对每个阈值统计 IoU>thr 的样本数
        # num_acc: (9,)
        # ---------------------------------------------------------
        B = ious.shape[0]
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0).to(torch.float32)
        accum_acc += num_acc
        accum_iou += ious.sum().to(torch.float32)
        accum_sample += float(B)

        # 保存预测（只在 rank0 做）
        if save_preds and _is_rank0():
            all_pred_ious.append(ious.view(-1, 1).detach().cpu())
            if ("orig_bbox" in tdict) and (postprocessors is not None):
                all_pred_boxes.append(pred_boxes.detach().cpu())
            else:
                all_pred_boxes.append(pred_xyxy.detach().cpu())

    total = time.time() - start

    # 同步 logger（loss 等）
    metric_logger.synchronize_between_processes()

    # 分布式下把 accum_* 做 all_reduce，得到全局统计
    if _is_dist():
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    denom = float(accum_sample.item()) if float(accum_sample.item()) > 0 else 1.0
    acc = (accum_acc / denom).detach().cpu()          # (9,)
    miou = float((accum_iou / denom).item())          # 平均 IoU

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': float(a.item()) for t, a in zip(iou_thrs.detach().cpu(), acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {"sec": float(total)}

    # 保存预测结果文件（只在 rank0）
    if save_preds and _is_rank0():
        try:
            torch.save(
                {
                    "pred_boxes": torch.cat(all_pred_boxes, dim=0) if len(all_pred_boxes) > 0 else torch.empty((0, 4)),
                    "pred_ious": torch.cat(all_pred_ious, dim=0) if len(all_pred_ious) > 0 else torch.empty((0, 1)),
                },
                save_pred_path + "pred_boxes"   # ⚠️注意这里不是 .pt 扩展名，但不影响 torch.save
            )
        except Exception as e:
            print(f"[WARN] save_pred_path failed: {e}")

    return val_stats, val_acc, val_time


def train_one_epoch_w_accum(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    max_norm: float = 0.0,
    args=None,
    writer=None,
    global_step_start: int = 0,
    tb_log_freq: int = 20,
    tb_vis_freq: int = 500,
    tb_num_vis: int = 2,
    postprocessor=None,
    tb_tag_prefix: str = "",
    debug_vis: bool = False,
    debug_vis_freq: int = 200,
    debug_vis_num: int = 8,
    debug_vis_dir: str = "outputs/debug_vis_train",
    **kwargs,
):
    """
    兼容 train.py：有些仓库会在 enable_batch_accum 时调用 train_one_epoch_w_accum。

    你这里不重复写逻辑，而是直接复用 train_one_epoch：
      - accum_iter 由 args.enable_batch_accum + args.accum_iter 控制
    """
    return train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        epochs=epochs,
        max_norm=max_norm,
        args=args,
        writer=writer,
        global_step_start=global_step_start,
        tb_log_freq=tb_log_freq,
        debug_vis=debug_vis,
        debug_vis_freq=debug_vis_freq,
        debug_vis_num=debug_vis_num,
        debug_vis_dir=debug_vis_dir,
        **kwargs,
    )
