import os
import time

# ====== 关键：无GUI环境用 Agg 后端（必须在 import pyplot 之前）======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import sam3
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor

# plot_results 依赖 sklearn；如果没装，自动降级只保存“画框图”
try:
    from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
    HAS_PLOT = True
except Exception as e:
    print("[WARN] visualization_utils import failed, will skip plot_results:", repr(e), flush=True)
    from sam3.visualization_utils import draw_box_on_image, normalize_bbox
    HAS_PLOT = False


def save_current_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[SAVE] {path}", flush=True)


def main():
    print("=== SAM3 offline GPU demo start ===", flush=True)
    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    print("torch.version.cuda:", torch.version.cuda, flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== 改这里：你的本地权重路径（sam3.pt）======
    ckpt_path = "/mnt/data_2/pl/miccai/PLVL_SAM3/sam3-main/checkpoint/sam3.pt"
    assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    print("ckpt_path:", ckpt_path, flush=True)

    # 资源路径
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    image_path = f"{sam3_root}/assets/images/test_image.jpg"
    assert os.path.isfile(image_path), f"image not found: {image_path}"

    out_dir = os.path.join(os.path.dirname(__file__), "outputs_demo")
    os.makedirs(out_dir, exist_ok=True)

    # TF32（4090 支持）
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ====== build + load (离线) ======
    t0 = time.time()
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        load_from_HF=False,   # <- 关键：禁止联网
        device=device,
        eval_mode=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"build+load time: {time.time()-t0:.3f}s", flush=True)
    print("model device:", next(model.parameters()).device, flush=True)

    processor = Sam3Processor(model, confidence_threshold=0.5)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # ====== set_image（最耗时的一步，应当上GPU、显存上涨）======
    t0 = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = processor.set_image(image)
        else:
            inference_state = processor.set_image(image)
    if device == "cuda":
        torch.cuda.synchronize()
        print("cuda mem (MB):", torch.cuda.memory_allocated() / 1024 / 1024, flush=True)
    print(f"set_image time: {time.time()-t0:.3f}s", flush=True)

    # 保存原图
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    save_current_fig(os.path.join(out_dir, "0_input.png"))

    # ============================================================
    # A) Text prompt
    # ============================================================
    t0 = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="shoe")
        else:
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt="shoe")
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"text prompt time: {time.time()-t0:.3f}s", flush=True)

    if HAS_PLOT:
        img0 = Image.open(image_path).convert("RGB")
        plot_results(img0, inference_state)
        save_current_fig(os.path.join(out_dir, "1_text_prompt_results.png"))
    else:
        print("[INFO] plot_results skipped (no sklearn).", flush=True)

    # ============================================================
    # B) Single box prompt
    # ============================================================
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    print("Normalized box:", norm_box_cxcywh, flush=True)

    t0 = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                processor.reset_all_prompts(inference_state)
                inference_state = processor.add_geometric_prompt(
                    state=inference_state, box=norm_box_cxcywh, label=True
                )
        else:
            processor.reset_all_prompts(inference_state)
            inference_state = processor.add_geometric_prompt(
                state=inference_state, box=norm_box_cxcywh, label=True
            )
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"single box prompt time: {time.time()-t0:.3f}s", flush=True)

    img0 = Image.open(image_path).convert("RGB")
    image_with_box = draw_box_on_image(img0, box_input_xywh.flatten().tolist())
    plt.figure()
    plt.imshow(image_with_box)
    plt.axis("off")
    save_current_fig(os.path.join(out_dir, "2_single_box.png"))

    if HAS_PLOT:
        plot_results(img0, inference_state)
        save_current_fig(os.path.join(out_dir, "3_single_box_results.png"))

    # ============================================================
    # C) Multi-box prompt
    # ============================================================
    box_input_xywh = [[480.0, 290.0, 110.0, 360.0], [370.0, 280.0, 115.0, 375.0]]
    box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_input_xywh).view(-1, 4))
    norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).tolist()
    box_labels = [True, False]

    t0 = time.time()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                processor.reset_all_prompts(inference_state)
                for box, label in zip(norm_boxes_cxcywh, box_labels):
                    inference_state = processor.add_geometric_prompt(
                        state=inference_state, box=box, label=label
                    )
        else:
            processor.reset_all_prompts(inference_state)
            for box, label in zip(norm_boxes_cxcywh, box_labels):
                inference_state = processor.add_geometric_prompt(
                    state=inference_state, box=box, label=label
                )
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"multi-box prompt time: {time.time()-t0:.3f}s", flush=True)

    # 画多框
    img0 = Image.open(image_path).convert("RGB")
    image_with_box = img0
    for b, lab in zip(box_input_xywh, box_labels):
        color = (0, 255, 0) if lab else (255, 0, 0)
        image_with_box = draw_box_on_image(image_with_box, b, color)

    plt.figure()
    plt.imshow(image_with_box)
    plt.axis("off")
    save_current_fig(os.path.join(out_dir, "4_multi_box.png"))

    if HAS_PLOT:
        plot_results(img0, inference_state)
        save_current_fig(os.path.join(out_dir, "5_multi_box_results.png"))

    print(f"=== DONE. Outputs saved to: {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
