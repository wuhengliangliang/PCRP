# -*- coding: utf-8 -*-
"""
V1 Ours（默认卖点）：Stage2 注入视觉 cue + learnable gate（初始弱注入）
目的：展示视觉 cue 在 refine 阶段“校正语言表征 -> 提升 grounding”的增益
"""

dataset = 'pathology2'
output_dir = 'outputs/pathology_ablation/V1_stage2_gate_sam3'

checkpoint_best = True
batch_size = 12
epochs = 120
lr_drop = 90

freeze_epochs = 3
freeze_modules = ['backbone']
load_weights_path = 'pretrained_checkpoints/detr-r50.pth'

debug_vis = True
debug_vis_freq = 1000
debug_vis_num = 2
debug_vis_dir = f"{output_dir}/debug_vis_train"

accum_iter = 2
enable_batch_accum = True

# ===== Fusion (ON, Stage2 only) =====
use_fusion = True
fusion_use_stage1 = False
fusion_use_stage2 = True

# gate_init=-4 => sigmoid≈0.018：训练初期几乎不注入，稳定后逐渐学会用 cue
fusion_gate_init = -4.0
fusion_nheads = 8
fusion_dropout = 0.1
# fusion_ffn_dim = None

# ===== SAM3 (ON) =====
use_sam3 = True
sam3_ckpt = '/mnt/data_2/pl/miccai/best_pathvgsam3/sam3/checkpoint/sam3.pt'
sam3_bpe_path = '/mnt/data_2/pl/miccai/best_pathvgsam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'

sam3_resolution = 1008
sam3_confidence_threshold = 0.8
sam3_prompt_coord = 'norm'

debug_sam3 = False
debug_sam3_every = 50

sam3_prompt_use_gt_when_training = True
sam3_prompt_warmup_epochs = 5
sam3_prompt_mix_epochs = 20

model_config = dict(
    decoder=dict(
        type='VisualDenstanglingPrototype',
        num_queries=1,
        query_dim=256,
        return_intermediate=True,
        num_extra_layers=1,
        extra_layer=dict(
            type='DiscriminativeFeatEncLayer',
            d_model=256,
            img_query_with_pos=False,
            img2text_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            discrimination_coef_settings=dict(
                text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                scale=1.0,
                sigma=0.5,
                pow=2.0,
            ),
        )
    )
)
