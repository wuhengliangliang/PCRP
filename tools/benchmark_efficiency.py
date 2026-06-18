#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAM3_PARENT = ROOT / "sam3"
if SAM3_PARENT.exists() and str(SAM3_PARENT) not in sys.path:
    sys.path.insert(0, str(SAM3_PARENT))

import eval as eval_mod  # noqa: E402
from models import build_model  # noqa: E402
from util.config import Config  # noqa: E402


def _cli_has(name: str) -> bool:
    prefix = f"--{name}"
    return any(a == prefix or a.startswith(prefix + "=") for a in sys.argv[1:])


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _count_params(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"params": int(total), "trainable_params": int(trainable)}


def _count_sam3_teacher_params(model: torch.nn.Module) -> int:
    teacher = getattr(model, "sam3", None)
    teacher_model = getattr(teacher, "model", None)
    if isinstance(teacher_model, torch.nn.Module):
        return int(sum(p.numel() for p in teacher_model.parameters()))
    return 0


def _load_weights(model: torch.nn.Module, resume: str, strict: bool = False) -> Dict[str, Any]:
    if not resume:
        return {"loaded": False, "missing": [], "unexpected": []}

    resume_path = Path(resume)
    if resume_path.name == "meta.pth" and resume_path.parent.exists():
        resume = str(resume_path.parent)

    ckpt = eval_mod.load_checkpoint_any(resume, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(sd, strict=bool(strict))
    return {
        "loaded": True,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def _make_text(batch_size: int, seq_len: int, device: torch.device):
    ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    # [CLS] visual target [SEP]. Token 5101 is a common BERT word id; exact text is
    # irrelevant for synthetic latency, but nonzero ids avoid all-padding masks.
    ids[:, 0] = 101
    ids[:, 1] = 5101
    ids[:, 2] = 4539
    ids[:, 3] = 102
    mask[:, :4] = True
    return ids, mask


def _make_inputs(args, device: torch.device):
    b = int(args.bench_batch_size)
    h = int(args.bench_image_size)
    w = int(args.bench_image_size)
    seq_len = int(args.bench_seq_len)

    image = torch.rand((b, 3, h, w), dtype=torch.float32, device=device)
    image_mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)

    word_id, word_mask = _make_text(b, seq_len, device)
    r0, rm0 = _make_text(b, seq_len, device)
    r1, rm1 = _make_text(b, seq_len, device)
    r2, rm2 = _make_text(b, seq_len, device)

    return (image, image_mask, word_id, word_mask, r0, rm0, r1, rm1, r2, rm2)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _amp_context(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.inference_mode()


def _forward(model: torch.nn.Module, inputs, use_amp: bool, device: torch.device):
    if use_amp and device.type == "cuda":
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(*inputs, epoch=0, step=0)
    with torch.inference_mode():
        return model(*inputs, epoch=0, step=0)


def _profile_flops(model: torch.nn.Module, inputs, use_amp: bool, device: torch.device) -> int | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as prof:
            _forward(model, inputs, use_amp, device)
            _sync(device)
        total = 0
        for evt in prof.key_averages():
            total += int(getattr(evt, "flops", 0) or 0)
        return int(total) if total > 0 else None
    except Exception as exc:
        print(f"[WARN] FLOPs profiling failed: {exc}", file=sys.stderr)
        return None


def benchmark(args) -> Dict[str, Any]:
    cfg = Config(args.config)
    cfg.merge_to_args(args)

    # eval.py defaults to an old path; use this repo's current best model unless
    # the caller explicitly passes --resume.
    if (not _cli_has("resume")) and (not Path(str(args.resume)).exists()):
        args.resume = str(ROOT / "outputs/pathology_reason_test/sam3_new_2/ckpts/best_joint/meta.pth")

    args.device = str(_device(str(args.device)))
    args.distributed = False
    args.rank = 0
    args.gpu = 0
    args.load_weights_path = getattr(args, "load_weights_path", "")
    eval_mod._ensure_sam3_alias(args)

    device = _device(str(args.device))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    load_info = _load_weights(model, str(args.resume), strict=bool(args.strict_load))
    inputs = _make_inputs(args, device)

    # Warm-up.
    for _ in range(int(args.warmup)):
        _forward(model, inputs, bool(args.amp), device)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(int(args.iters)):
        _forward(model, inputs, bool(args.amp), device)
    _sync(device)
    elapsed = time.perf_counter() - t0

    sec_per_iter = elapsed / max(1, int(args.iters))
    batch_size = int(args.bench_batch_size)
    result: Dict[str, Any] = {
        "config": str(args.config),
        "resume": str(args.resume),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "use_sam3": bool(getattr(args, "use_sam3", False)),
        "amp": bool(args.amp),
        "batch_size": batch_size,
        "image_size": int(args.bench_image_size),
        "seq_len": int(args.bench_seq_len),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "sec_per_iter": sec_per_iter,
        "ms_per_image": sec_per_iter * 1000.0 / max(1, batch_size),
        "fps": batch_size / sec_per_iter,
        "peak_mem_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda"
            else None
        ),
        "model_params": _count_params(model),
        "sam3_teacher_params": _count_sam3_teacher_params(model),
        "checkpoint_loaded": bool(load_info["loaded"]),
        "missing_keys": len(load_info["missing"]),
        "unexpected_keys": len(load_info["unexpected"]),
    }

    if bool(args.profile_flops):
        result["profiler_flops"] = _profile_flops(model, inputs, bool(args.amp), device)

    return result


def main():
    parser = argparse.ArgumentParser(
        "PathVG efficiency benchmark",
        parents=[eval_mod.get_args_parser()],
    )
    parser.add_argument("--bench_batch_size", default=1, type=int)
    parser.add_argument("--bench_image_size", default=768, type=int)
    parser.add_argument("--bench_seq_len", default=40, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iters", default=50, type=int)
    parser.add_argument("--amp", default=True, type=eval_mod.boolean_string)
    parser.add_argument("--profile_flops", default=False, type=eval_mod.boolean_string)
    parser.add_argument("--output_json", default="", type=str)
    args = parser.parse_args()

    result = benchmark(args)

    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print("| use_sam3 | batch | image | ms/img | FPS | peak mem MB | model params | SAM3 teacher params |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    peak = result["peak_mem_mb"]
    peak_text = f"{peak:.1f}" if peak is not None else "NA"
    print(
        f"| {result['use_sam3']} | {result['batch_size']} | {result['image_size']} | "
        f"{result['ms_per_image']:.2f} | {result['fps']:.2f} | "
        f"{peak_text} | "
        f"{result['model_params']['params']} | {result['sam3_teacher_params']} |"
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
