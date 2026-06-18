#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _load_ious(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "pred_ious" not in obj:
        raise RuntimeError(f"{path} must be a torch-save dict with key 'pred_ious'")
    ious = obj["pred_ious"]
    if torch.is_tensor(ious):
        ious = ious.detach().cpu().view(-1).float().numpy()
    else:
        ious = np.asarray(ious, dtype=np.float32).reshape(-1)
    return ious.astype(np.float64, copy=False)


def paired_bootstrap(
    baseline_ious: np.ndarray,
    ours_ious: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    if baseline_ious.shape != ours_ious.shape:
        raise ValueError(f"shape mismatch: baseline={baseline_ious.shape}, ours={ours_ious.shape}")

    n = int(baseline_ious.size)
    if n <= 0:
        raise ValueError("empty IoU arrays")

    base_hit = (baseline_ious > threshold).astype(np.float64)
    ours_hit = (ours_ious > threshold).astype(np.float64)
    hit_delta = ours_hit - base_hit
    iou_delta = ours_ious - baseline_ious

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n), endpoint=False)
    boot_hit = hit_delta[idx].mean(axis=1)
    boot_iou = iou_delta[idx].mean(axis=1)

    hit_delta_mean = float(hit_delta.mean())
    iou_delta_mean = float(iou_delta.mean())

    p_hit = float(2.0 * min(np.mean(boot_hit <= 0.0), np.mean(boot_hit >= 0.0)))
    p_iou = float(2.0 * min(np.mean(boot_iou <= 0.0), np.mean(boot_iou >= 0.0)))

    return {
        "n": n,
        "threshold": float(threshold),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "baseline_acc": float(base_hit.mean()),
        "ours_acc": float(ours_hit.mean()),
        "delta_acc": hit_delta_mean,
        "delta_acc_ci95": [float(x) for x in np.percentile(boot_hit, [2.5, 97.5])],
        "delta_acc_p_two_sided": min(1.0, p_hit),
        "baseline_mean_iou": float(baseline_ious.mean()),
        "ours_mean_iou": float(ours_ious.mean()),
        "delta_mean_iou": iou_delta_mean,
        "delta_mean_iou_ci95": [float(x) for x in np.percentile(boot_iou, [2.5, 97.5])],
        "delta_mean_iou_p_two_sided": min(1.0, p_iou),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    baseline_ious = _load_ious(args.baseline)
    ours_ious = _load_ious(args.ours)
    result = paired_bootstrap(
        baseline_ious=baseline_ious,
        ours_ious=ours_ious,
        threshold=float(args.threshold),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
