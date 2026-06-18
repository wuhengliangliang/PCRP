# -*- coding: utf-8 -*-
"""
S0：禁用 SAM3（teacher prior 关），只保留我们 Stage2 的视觉 cue 注入
目的：把“视觉 cue 的贡献”从 teacher prior 里剥离出来，证明 cue 本身有效（尤其 hard）
"""

dataset = 'pathology2'
output_dir = 'outputs/pathology_ablation/S0_stage2_gate_no_sam3'

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
fusion_gate_init = -4.0
fusion_nheads = 8
fusion_dropout = 0.1

# ===== SAM3 (OFF) =====
use_sam3 = False

# 这些参数留着无所谓（不会用到），但保留便于你复制对齐
sam3_prompt_use_gt_when_training = True
sam3_prompt_warmup_epochs = 5
sam3_prompt_mix_epochs = 20
debug_sam3 = False
debug_sam3_every = 50

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
