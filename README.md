# PCRP

<h3 align="center">
PCRP: Progressive Coarse-to-Fine Refinement for Pathology Grounding
</h3>

<p align="center">
  <strong>MICCAI 2026</strong>
</p>

<p align="center">
  Liang Peng<sup>1,*</sup>, Chenxiao Li<sup>2,*</sup>, Shaohua Dong<sup>2</sup>,
  Bohan Tan<sup>3</sup>, Zhipeng Zhang<sup>4</sup>, Xingping Dong<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Wuhan University &nbsp;&nbsp;
  <sup>2</sup>University of North Texas &nbsp;&nbsp;
  <sup>3</sup>Huazhong University of Science and Technology &nbsp;&nbsp;
  <sup>4</sup>SAI, Shanghai Jiao Tong University
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#installation">Installation</a> |
  <a href="#data-and-weights">Data & Weights</a> |
  <a href="#training">Training</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#results">Results</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="asset/PCRP.pdf" width="95%" alt="PCRP framework overview">
</p>

## Overview

PCRP is a pathology visual grounding framework for localizing phrase-described
pathological regions in histopathology images. Pathology grounding is challenging
because target regions can be small, low-contrast, and boundary-ambiguous, while
medical expressions are often short and highly specialized.

To address these issues, PCRP follows a progressive coarse-to-fine design. It
first predicts a stable coarse grounding box, then uses the coarse box to obtain
reliable visual cues from a frozen SAM3 mask prior, and finally performs
boundary-aware Stage-2 refinement with language-conditioned visual features.
This design reduces semantic drift and boundary drift on difficult samples such
as tiny targets, weak boundaries, and low-magnification pathology images.

This repository contains the runnable code used for the PCRP experiments on
RefPath/PathVG-style pathology visual grounding. The implementation is built on
the TransCP/PathVG codebase and adds:

- Visual-Word Fusion (VWF) for word-wise visual evidence retrieval and gated
  language-feature refinement.
- Coarse-to-Fine Boundary-Aware Regression (C2F-BAR) for progressive Stage-1
  localization and Stage-2 boundary refinement.
- Confidence-weighted Boundary Prior (CBP) from top-K SAM3 candidate masks.
- Progressive Prompt Transition (PPT), which gradually shifts SAM3 prompts from
  ground-truth boxes to predicted coarse boxes during training.
- Sharded checkpoint saving/loading for distributed training and evaluation.

## News

- PCRP has been accepted to MICCAI 2026.
- The current public code snapshot includes training, evaluation, SAM3 prior
  integration, ablation configs, and the framework figure.
- Pretrained PCRP checkpoints and additional dataset/resource links will be
  updated when release permissions are finalized.

## Method

The main model is implemented in [models/transcp.py](models/transcp.py). The
forward path is:

1. Encode the pathology image with a ResNet-50 backbone and visual transformer.
2. Encode the referring expression with the BERT text branch.
3. Apply Visual-Word Fusion (VWF), where text tokens query visual tokens through
   cross-attention and receive gated visual evidence.
4. Predict a coarse Stage-1 grounding box.
5. Prompt SAM3 with the coarse box to obtain top-K candidate mask priors.
6. Convert the confidence-weighted SAM3 mask prior into token-level bias with
   [models/mask_prior_adapter.py](models/mask_prior_adapter.py).
7. Run Stage-2 boundary-aware visual-language refinement and output the final
   grounding box.

During training, PCRP uses Progressive Prompt Transition (PPT): SAM3 is first
prompted with ground-truth boxes for warmup and then gradually transitions to
predicted Stage-1 boxes. During inference, SAM3 is prompted only by the
predicted coarse box.

## Qualitative Example

The following examples are referenced with repository-relative paths so they
render correctly on GitHub.

<p align="center">
  <img src="sam3/outputs_pathvg_sam3_box/8006_bbox1_box.png" width="45%" alt="SAM3 box-prompt visual cue example">
  <img src="sam3/outputs_pathvg_sam3_text/8006_bbox1_text.png" width="45%" alt="SAM3 text-prompt visual cue example">
</p>

## Repository Layout

```text
.
├── train.py                         # main training entry
├── eval.py                          # evaluation entry; supports sharded checkpoints
├── engine.py                        # training and evaluation loops
├── configs/
│   ├── TransCP_R50_pathology2.py    # main PCRP/SAM3-assisted config
│   └── TransCP_R50_pathology2_*.py  # ablation configs
├── models/
│   ├── transcp.py                   # PCRP model implementation
│   ├── sam3_wrapper.py              # SAM3 box-prompt teacher wrapper
│   └── mask_prior_adapter.py        # mask prior to token-bias adapter
├── datasets/
│   └── dataset.py                   # RefPath/pathology2 dataset loader
├── split/data/pathology2/           # prepared split files
├── pretrained_checkpoints/          # DETR, ResNet-50, and BERT initialization
├── sam3/
│   ├── sam3/                        # bundled SAM3 code
│   ├── assets/bpe_simple_vocab_16e6.txt.gz
│   └── checkpoint/sam3.pt           # SAM3 checkpoint
├── asset/
│   └── model.png                    # PCRP framework figure used in this README
└── outputs/                         # local training/evaluation outputs
```

## Installation

Create a Python environment and install the project dependencies:

```bash
conda create -n pcrp python=3.8 -y
conda activate pcrp

# Install the PyTorch/CUDA version that matches your machine first.
# Example only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

The local `sam3/` directory is added to `sys.path` by `train.py` and `eval.py`.
If you run custom scripts, export it manually:

```bash
export PYTHONPATH="$PWD/sam3:$PYTHONPATH"
```

## Data and Weights

### Dataset

PCRP is evaluated on RefPath. Following the PathVG setting, RefPath contains
27,610 histopathology images and 33,500 language-grounded bounding boxes. The
standard split contains 24,757 training images with 30,452 boxes and 2,853 test
images with 3,048 boxes. The test set is divided into `testA` for 40× images
and `testB` for 20× images.

The dataset loader expects `pathology2` split files under:

```text
split/data/pathology2/
├── refpath_train.pth
├── refpath_testA.pth
└── refpath_testB.pth
```

Images should be arranged under a data root that contains either
`pathology2/images/` or compatible image paths stored in the split files:

```text
<DATA_ROOT>/
└── pathology2/
    └── images/
```

When running on a new machine, pass `--data_root` and `--split_root` explicitly:

```bash
DATA_ROOT=/path/to/refpath_or_pathology_root
SPLIT_ROOT=$PWD/split/data
```

### Pretrained Initialization

Place backbone and language initialization files as follows:

```text
pretrained_checkpoints/
├── detr-r50.pth
├── resnet50-19c8e357.pth
├── bert-base-uncased.tar.gz
├── bert-base-uncased/
└── bert_base_uncased/
```

### SAM3 Assets

PCRP uses SAM3 as an external visual prior generator. The SAM3 assets are:

```text
sam3/checkpoint/sam3.pt
sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

`sam3.pt` is not the final PCRP grounding checkpoint. It is used to produce the
mask prior for Stage-2 refinement. PCRP grounding checkpoints are saved under
`outputs/.../ckpts/`.

## Training

The main config is [configs/TransCP_R50_pathology2.py](configs/TransCP_R50_pathology2.py).
It enables SAM3, sets image size to 768, and saves outputs to
`outputs/pathology_reason_test/sam3_new_2`.

Single-node multi-GPU training:

```bash
DATA_ROOT=/path/to/refpath_or_pathology_root
SPLIT_ROOT=$PWD/split/data

torchrun --nproc_per_node=2 --master_port=29516 train.py \
  --config configs/TransCP_R50_pathology2.py \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --sam3_ckpt "$PWD/sam3/checkpoint/sam3.pt" \
  --sam3_bpe_path "$PWD/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
```

Single-GPU debug run:

```bash
python train.py \
  --config configs/TransCP_R50_pathology2.py \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --sam3_ckpt "$PWD/sam3/checkpoint/sam3.pt" \
  --sam3_bpe_path "$PWD/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --batch_size 4 \
  --batch_size_test 1 \
  --num_workers 2
```

Resume from a sharded checkpoint directory:

```bash
torchrun --nproc_per_node=2 --master_port=29516 train.py \
  --config configs/TransCP_R50_pathology2.py \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --sam3_ckpt "$PWD/sam3/checkpoint/sam3.pt" \
  --sam3_bpe_path "$PWD/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --resume outputs/pathology_reason_test/sam3_new_2/ckpts/last
```

Training logs and checkpoints are written to:

```text
outputs/pathology_reason_test/sam3_new_2/
├── epoch.log
├── iter.txt
├── debug_vis_train/
└── ckpts/
    ├── best_joint/
    ├── best_testA/
    ├── best_testB/
    └── last/
```

Checkpoint directories contain sharded model weights:

```text
ckpts/best_joint/
├── meta.pth
├── rank000.pth
└── rank001.pth
```

Pass the checkpoint directory to `eval.py`, not only `meta.pth`.

## Evaluation

Evaluate `testA`:

```bash
DATA_ROOT=/path/to/refpath_or_pathology_root
SPLIT_ROOT=$PWD/split/data

torchrun --nproc_per_node=2 --master_port=29517 eval.py \
  --config configs/TransCP_R50_pathology2.py \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --test_split testA \
  --resume outputs/pathology_reason_test/sam3_new_2/ckpts/best_joint \
  --sam3_ckpt "$PWD/sam3/checkpoint/sam3.pt" \
  --sam3_bpe_path "$PWD/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --save_pred_path predictions_testA_
```

Evaluate `testB`:

```bash
torchrun --nproc_per_node=2 --master_port=29518 eval.py \
  --config configs/TransCP_R50_pathology2.py \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --test_split testB \
  --resume outputs/pathology_reason_test/sam3_new_2/ckpts/best_joint \
  --sam3_ckpt "$PWD/sam3/checkpoint/sam3.pt" \
  --sam3_bpe_path "$PWD/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --save_pred_path predictions_testB_
```

Evaluation outputs are saved under:

```text
<output_dir>/eval_results/
├── eval.log
├── eval_results.json
└── predictions_*_pred_boxes
```

## Results

Following RefPath, accuracy is computed with IoU threshold 0.70 for 40× images
(`testA`) and 0.50 for 20× images (`testB`). mIoU is also reported to measure
overall grounding quality.

### Main Results on RefPath

| Model | Venue | Visual/Text Encoder | RefPath-all Acc | RefPath-all mIoU | RefPath-40× Acc | RefPath-40× mIoU | RefPath-20× Acc | RefPath-20× mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TransVG | ICCV'21 | RN50/BERT-B | 58.40 | 52.86 | 68.72 | 66.75 | 50.29 | 41.94 |
| SeqTR | ECCV'22 | DN53/BiGRU | 55.84 | 51.96 | 72.65 | 71.13 | 42.57 | 36.78 |
| CLIPVG | TMM'23 | CLIP-B/CLIP-B | 58.89 | 53.97 | 75.52 | 72.14 | 45.81 | 39.67 |
| LLaVa-Med | NeurIPS'23 | CLIP-L/LLaMa | 62.32 | 57.96 | 73.52 | 70.24 | 53.51 | 48.31 |
| TransCP | TPAMI'24 | RN50/BERT-B | 61.73 | 56.81 | 74.27 | 71.92 | 51.87 | 44.93 |
| SimVG | NeurIPS'24 | ViT-B/BERT-B | 63.94 | 59.42 | 75.36 | 73.18 | 52.52 | 46.92 |
| D-MDETR | TPAMI'24 | CLIP-B/CLIP-B | 64.92 | 57.69 | 76.29 | 73.10 | 55.98 | 45.57 |
| PKNet | MICCAI'25 | RN50/BERT-B | 69.95 | 63.49 | 80.48 | 76.88 | 61.66 | 52.95 |
| **PCRP (Ours)** | **MICCAI'26** | **RN50/BERT-B** | **81.63** | **72.76** | **85.04** | **80.23** | **78.22** | **66.89** |

Compared with PKNet, PCRP improves RefPath-all by 11.68 Acc and 9.27 mIoU,
and improves RefPath-20× by 16.56 Acc and 13.94 mIoU.

### Ablation Study

| ID | VWF | CBP | PPT | RefPath-all Acc/mIoU | RefPath-40× Acc/mIoU | RefPath-20× Acc/mIoU |
|---|:---:|:---:|:---:|---:|---:|---:|
| (a) |  |  |  | 78.90 / 69.40 | 82.40 / 78.20 | 75.30 / 63.40 |
| (b) | ✓ |  |  | 79.85 / 70.20 | 82.95 / 78.55 | 76.90 / 65.10 |
| (c) | ✓ | ✓ |  | 81.30 / 72.45 | 84.55 / 80.10 | 78.15 / 66.70 |
| (d) | ✓ | ✓ | ✓ | **81.63 / 72.76** | **85.04 / 80.23** | **78.22 / 66.89** |

## Main Configs

| Config | Purpose |
|---|---|
| `configs/TransCP_R50_pathology2.py` | Main PCRP/SAM3-assisted model |
| `configs/TransCP_R50_pathology2_S0_stage2_gate_no_sam3.py` | Stage-2 cue setting without SAM3 prior |
| `configs/TransCP_R50_pathology2_S1_stage2_gate_sam3_no_gt_prompt.py` | SAM3 enabled, no GT prompt during training |
| `configs/TransCP_R50_pathology2_S2_stage2_gate_sam3_high_th.py` | SAM3 enabled with stricter confidence threshold |
| `configs/TransCP_R50_pathology2_V1_stage2_gate_sam3.py` | Stage-2 gated SAM3 cue |
| `configs/TransCP_R50_pathology2_V2_stage2_strong_inject_sam3.py` | Stronger cue injection |
| `configs/TransCP_R50_pathology2_V3_stage2_very_weak_gate_sam3.py` | Very weak cue injection |
| `configs/TransCP_R50_pathology2_V4_stage1_gate_sam3.py` | Stage-1 cue injection only |
| `configs/TransCP_R50_pathology2_V5_both_stage_gate_sam3.py` | Cue injection in both stages |

## Troubleshooting

- If SAM3 import fails, check that `sam3/` exists and run
  `export PYTHONPATH="$PWD/sam3:$PYTHONPATH"`.
- If SAM3 loading fails, verify `sam3/checkpoint/sam3.pt` and
  `sam3/assets/bpe_simple_vocab_16e6.txt.gz`.
- If image loading fails, check `--data_root`, `--split_root`, and the image
  paths saved in `split/data/pathology2/refpath_*.pth`.
- If evaluation cannot load weights, pass the checkpoint directory containing
  `meta.pth`, `rank000.pth`, and `rank001.pth`.
- If distributed launch reports a port conflict, change `--master_port`.

## Citation

If you use this repository, please cite our MICCAI 2026 paper:

```bibtex
@inproceedings{peng2026pcrp,
  title     = {PCRP: Progressive Coarse-to-Fine Refinement for Pathology Grounding},
  author    = {Peng, Liang and Li, Chenxiao and Dong, Shaohua and Tan, Bohan and Zhang, Zhipeng and Dong, Xingping},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer},
  note      = {To appear}
}
```

## Acknowledgements

This work was supported in part by the National Key Research and Development
Program of China under Grants 2023YFC2705704 and 2023YFC2705705, the
Fundamental Research Funds for the Central Universities (No. 2042026kf0044),
the National Natural Science Foundation of China (Grant No. 62471342), the
Innovative Research Group Project of Hubei Province under Grant 2024AFA017,
the New Cornerstone Science Foundation through the XPLORER PRIZE, and the
WHU-Kingsoft Joint Lab.

This codebase builds on DETR, TransVG/TransCP, PathVG, and the bundled SAM3
implementation. We thank the authors for releasing their code and resources.
