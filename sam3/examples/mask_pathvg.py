#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict

# ========= 无GUI服务器环境必须用 Agg =========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image, ImageDraw

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh

# plot_results 依赖 sklearn；没有就自动降级（只保存GT框图）
try:
    from sam3.visualization_utils import normalize_bbox, plot_results
    HAS_PLOT = True
except Exception as e:
    print("[WARN] plot_results not available (maybe no sklearn). Will save GT-box only.")
    print("[WARN] import error:", repr(e))
    from sam3.visualization_utils import normalize_bbox
    HAS_PLOT = False


# -----------------------------
# IO: json / jsonl
# -----------------------------
def read_json_or_jsonl(path: str):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for x in obj:
                yield x
        else:
            raise ValueError("JSON file must be a list or use .jsonl format")


# -----------------------------
# BBox utilities
# -----------------------------
def clamp_xyxy_for_prompt(x1, y1, x2, y2, W, H):
    """
    给“提示框 prompt”用：允许 x2==W, y2==H（很多数据集这么标）
    约束到 [0,W] / [0,H]，并保证 w/h >= 1
    """
    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)

    # clamp 到 [0, W] / [0, H]
    x1 = max(0.0, min(x1, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))

    # ensure order
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # xywh（这里按“右下角为边界坐标”的常见写法：w=x2-x1, h=y2-y1）
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    # 如果 x1==W 或 y1==H，w/h 会变成1但实际无意义；这类框本身就有问题，仍然做个最小可用修正
    x1 = min(x1, float(W - 1)) if W > 1 else 0.0
    y1 = min(y1, float(H - 1)) if H > 1 else 0.0
    # 再次确保不超过边界
    w = min(w, float(W) - x1) if W > 0 else w
    h = min(h, float(H) - y1) if H > 0 else h
    w = max(1.0, w)
    h = max(1.0, h)
    return x1, y1, w, h


def clamp_xyxy_for_draw(x1, y1, x2, y2, W, H):
    """
    给“画框”用：像素索引必须在 [0, W-1]/[0, H-1]，否则会越界
    并把 (x2,y2) 当成右下角边界坐标，压到 W-1/H-1。
    """
    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)

    x1 = max(0.0, min(x1, float(W - 1)))
    y1 = max(0.0, min(y1, float(H - 1)))
    x2 = max(0.0, min(x2, float(W - 1)))
    y2 = max(0.0, min(y2, float(H - 1)))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # 至少画一个像素宽高
    if x2 == x1 and W > 1:
        x2 = min(float(W - 1), x1 + 1.0)
    if y2 == y1 and H > 1:
        y2 = min(float(H - 1), y1 + 1.0)
    return x1, y1, x2, y2


def safe_draw_box_xyxy(img: Image.Image, xyxy, color=(0, 255, 0), width=3):
    W, H = img.size
    x1, y1, x2, y2 = clamp_xyxy_for_draw(xyxy[0], xyxy[1], xyxy[2], xyxy[3], W, H)
    im2 = img.copy()
    draw = ImageDraw.Draw(im2)
    draw.rectangle([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                   outline=color, width=width)
    return im2


def resolve_image_path(data_root: str, rel_or_abs: str):
    """
    兼容两种 data_root：
    1) data_root=/.../refpath_image/  && rel="refpath_image/xxx.jpg"
    2) data_root=/.../miccai/         && rel="refpath_image/xxx.jpg"
    3) rel="xxx.jpg" 也支持
    """
    # 绝对路径直接返回
    if os.path.isabs(rel_or_abs) and os.path.isfile(rel_or_abs):
        return rel_or_abs

    # 尝试 data_root + rel
    p1 = os.path.join(data_root, rel_or_abs)
    if os.path.isfile(p1):
        return p1

    # 如果 rel 带 refpath_image/ 前缀，但 data_root 本身已经是 refpath_image/
    if rel_or_abs.startswith("refpath_image" + os.sep) or rel_or_abs.startswith("refpath_image/"):
        rel2 = rel_or_abs.split("refpath_image/")[-1].split("refpath_image\\")[-1]
        p2 = os.path.join(data_root, rel2)
        if os.path.isfile(p2):
            return p2

    # 再尝试：如果 data_root 是 refpath_image/ 的父目录
    parent = os.path.dirname(data_root.rstrip("/"))
    p3 = os.path.join(parent, rel_or_abs)
    if os.path.isfile(p3):
        return p3

    return None


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[SAVE] {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="PathVG annotation file (.jsonl or .json)")
    parser.add_argument("--data_root", required=True, help="Root folder for images (can be .../refpath_image/ or its parent)")
    parser.add_argument("--ckpt", required=True, help="Local SAM3 checkpoint path (sam3.pt)")
    parser.add_argument("--out", default="outputs_pathvg_sam3", help="Output directory for PNGs")
    parser.add_argument("--mode", default="text", choices=["text", "box", "text+box"])
    parser.add_argument("--conf", type=float, default=0.5, help="Sam3Processor confidence_threshold")
    parser.add_argument("--max_images", type=int, default=-1, help="Process at most N unique images")
    parser.add_argument("--max_anns", type=int, default=-1, help="Process at most N annotations total")
    args = parser.parse_args()

    assert os.path.isfile(args.ckpt), f"ckpt not found: {args.ckpt}"
    assert os.path.isfile(args.ann), f"ann not found: {args.ann}"
    assert os.path.isdir(args.data_root), f"data_root not found: {args.data_root}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bpe
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    # build (离线本地 ckpt)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.ckpt,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, confidence_threshold=args.conf)

    # 读标注并按 image 分组（提速关键）
    groups = defaultdict(list)
    ann_loaded = 0
    for ann in read_json_or_jsonl(args.ann):
        rel = ann.get("image", None)
        if not rel:
            # fallback
            rel = ann.get("image_id", None)
            if rel:
                rel = os.path.join("refpath_image", rel)
        if not rel:
            continue
        groups[rel].append(ann)
        ann_loaded += 1
        if args.max_anns > 0 and ann_loaded >= args.max_anns:
            break

    img_list = list(groups.keys())
    if args.max_images > 0:
        img_list = img_list[: args.max_images]

    os.makedirs(args.out, exist_ok=True)
    print(f"total images: {len(img_list)}  (anns loaded: {ann_loaded})", flush=True)

    processed = 0

    for idx, rel in enumerate(img_list):
        img_path = resolve_image_path(args.data_root, rel)
        if img_path is None:
            print(f"[SKIP] missing image for rel={rel}", flush=True)
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # ---- 最慢：一张图只做一次 set_image ----
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    state = processor.set_image(img)
            else:
                state = processor.set_image(img)

        if device == "cuda" and idx == 0:
            torch.cuda.synchronize()
            print("cuda mem (MB):", torch.cuda.memory_allocated() / 1024 / 1024, flush=True)

        # 遍历该图所有标注
        for ann in groups[rel]:
            bbox_id = ann.get("bbox_id", "NA")
            expr_list = ann.get("expression", [""])
            expr = " ".join(expr_list).strip()

            gt_xyxy = ann.get("bbox", None)
            if gt_xyxy is None or len(gt_xyxy) != 4:
                continue

            # 先画 GT 框（永不越界）
            base = safe_draw_box_xyxy(img, gt_xyxy, color=(0, 255, 0), width=3)

            # prompt 推理
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        processor.reset_all_prompts(state)
                        st = state

                        if args.mode in ["text", "text+box"]:
                            st = processor.set_text_prompt(state=st, prompt=expr)

                        if args.mode in ["box", "text+box"]:
                            # 用 GT bbox 作为几何 prompt（xyxy -> xywh -> cxcywh -> normalize）
                            x1, y1, w, h = clamp_xyxy_for_prompt(
                                gt_xyxy[0], gt_xyxy[1], gt_xyxy[2], gt_xyxy[3], W, H
                            )
                            box_xywh = torch.tensor([x1, y1, w, h], dtype=torch.float32).view(-1, 4)
                            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                            norm = normalize_bbox(box_cxcywh, W, H).flatten().tolist()
                            st = processor.add_geometric_prompt(state=st, box=norm, label=True)

                else:
                    processor.reset_all_prompts(state)
                    st = state

                    if args.mode in ["text", "text+box"]:
                        st = processor.set_text_prompt(state=st, prompt=expr)

                    if args.mode in ["box", "text+box"]:
                        x1, y1, w, h = clamp_xyxy_for_prompt(
                            gt_xyxy[0], gt_xyxy[1], gt_xyxy[2], gt_xyxy[3], W, H
                        )
                        box_xywh = torch.tensor([x1, y1, w, h], dtype=torch.float32).view(-1, 4)
                        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
                        norm = normalize_bbox(box_cxcywh, W, H).flatten().tolist()
                        st = processor.add_geometric_prompt(state=st, box=norm, label=True)

            # 保存结果
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{stem}_bbox{bbox_id}_{args.mode}.png"
            out_path = os.path.join(args.out, out_name)

            if HAS_PLOT:
                # plot_results 会打印 found N object(s)
                plot_results(base, st)
                save_fig(out_path)
            else:
                # 没有 sklearn：仅保存 GT 框图（至少能跑通&检查数据）
                plt.figure()
                plt.imshow(base)
                plt.axis("off")
                save_fig(out_path)

            processed += 1

    print(f"DONE. processed={processed}, saved to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
