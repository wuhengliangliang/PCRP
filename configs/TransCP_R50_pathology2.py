# configs/TransCP_R50_pathology2.py
# -*- coding: utf-8 -*-

dataset = 'pathology2'

output_dir = 'outputs/pathology_reason_test/sam3_new_2'

checkpoint_best = True
batch_size = 64
epochs = 120

# ✅ 建议：drop 晚一点
lr_drop = 90

# ✅ 建议：backbone 冻结别太久（病理域适配很关键）
freeze_epochs = 3
freeze_modules = ['backbone']

load_weights_path = 'pretrained_checkpoints/detr-r50.pth'

debug_vis = True
debug_vis_freq = 1000
debug_vis_num = 2
debug_vis_dir = "outputs/pathology_reason_test/sam3_new_2/debug_vis_train"

debug_sam3_every = 50

# 训练早期用 GT prompt 给 SAM
sam3_prompt_use_gt_when_training = True
sam3_prompt_warmup_epochs = 5
sam3_prompt_mix_epochs = 20   # ✅ 建议：更平滑过渡


# ===================== ✅ Gradient Accumulation =====================
# 等效 batch = batch_size * accum_iter * world_size
# accum_iter = 2
# enable_batch_accum = True


# ============ ✅ SAM3 Teacher-in-the-loop settings ============
use_sam3 = True

sam3_ckpt = '/mnt/data_2/pl/miccai/best_pathvgsam3/sam3/checkpoint/sam3.pt'
sam3_bpe_path = '/mnt/data_2/pl/miccai/best_pathvgsam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'

sam3_resolution = 1008
sam3_confidence_threshold = 0.8   # ✅ 建议：降低一点，提高 teacher 触发率
sam3_prompt_coord = 'norm'
# ============================================================

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
