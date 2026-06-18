# models/transcp.py
# -*- coding: utf-8 -*-

"""
TransCP 主文件：model + criterion + postprocess

你引入的核心思路：
    guided_word_feat = fusion(img_tokens, word_tokens)
    然后 bbox_regression / refine 使用 guided_word_feat

为什么要这样做？
- 原 TransCP 的跨模态对齐依赖 “word token 表征” + “视觉 token 表征” 的 cross-attn / decoder 交互。
- 但是病理图像里，同一个词（如 cell / nucleus）对应的视觉区域极小、纹理相似，语言 token 容易“泛化”，导致对齐不够尖锐。
- 所以我们先用 image tokens 反向“校正” word tokens（word <- attend(image)），再喂给 bbox head。
- 为了稳定训练，引入 learnable gate（初始接近 0），避免一开始就破坏原本收敛路径。

⚠️ MultiheadAttention 的 key_padding_mask 只会 mask K/V，不会 mask Query（word）。
因此必须手动把 padded word query 的输出置零，否则会把噪声传播进后续层。
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from typing import Optional

from util import box_ops
from util.misc import (NestedTensor, get_world_size, is_dist_avail_and_initialized)
from pytorch_pretrained_bert.modeling import BertModel

# 图像 backbone（ResNet + position encoding）
from .backbone import build_backbone
# 视觉 transformer encoder（把 2D feature -> token，并输出 mask/pos）
from .transformer import build_visual_encoder
# bbox regression（transcp_visual 里：两阶段 decoder）
from .transcp_visual import build_bbox_regression
# 文本模块（BERT + GRU + phrase attention）
from .transcp_text import build_LSTMBert

# SAM3 teacher + mask prior adapter
from .sam3_wrapper import Sam3BoxMaskTeacher
from .mask_prior_adapter import MaskPriorAdapter


class WordGuidedFusion(nn.Module):
    """
    WordGuidedFusion：用视觉 token 引导 word token（word <- attend(image)）

    输入:
      img_tokens: (S, B, C)   S = HW，视觉 token
      word_tokens: (L, B, C)  L = 词序列长度
      word_pad_mask: (B, L)   True=padding（注意：MHA 不会 mask Query）
      img_pad_mask: (B, S)    True=padding（作为 key_padding_mask）

    输出:
      guided_word_tokens: (L, B, C)

    关键点：
    1) cross-attn：Q=word, K/V=image，让每个词从全图 token 中“挑选”与自己相关的视觉区域信息
    2) learnable gate：alpha 初始为负，使 sigmoid(alpha)≈0 -> 初期几乎不注入，训练更稳
    3) query padding 清零：因为 MultiheadAttention 不会 mask Query，必须手动处理 padded word token
    """

    def __init__(
        self,
        dim: int,
        nheads: int = 8,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
        gate_init: float = -4.0,  # sigmoid(-4)≈0.018，初始几乎不注入（稳定关键）
    ):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.dropout = float(dropout)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else int(dim * 4)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=nheads,
            dropout=self.dropout,
            batch_first=False
        )

        self.alpha = nn.Parameter(torch.tensor(float(gate_init)))

        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(self.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, self.ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(self.dropout)

    def forward(
        self,
        img_tokens: torch.Tensor,
        word_tokens: torch.Tensor,
        word_pad_mask: Optional[torch.Tensor] = None,
        img_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if word_pad_mask is not None:
            word_pad_mask = word_pad_mask.bool()  # (B,L)
        if img_pad_mask is not None:
            img_pad_mask = img_pad_mask.bool()    # (B,S)

        attn_out, _ = self.cross_attn(
            query=word_tokens,      # (L,B,C)
            key=img_tokens,         # (S,B,C)
            value=img_tokens,       # (S,B,C)
            key_padding_mask=img_pad_mask
        )

        gate = torch.sigmoid(self.alpha)
        x = word_tokens + gate * self.drop1(attn_out)
        x = self.norm1(x)

        x = x + self.drop2(self.ffn(x))
        x = self.norm2(x)

        if word_pad_mask is not None:
            pad_q = word_pad_mask.transpose(0, 1).unsqueeze(-1)  # (L,B,1)
            x = x.masked_fill(pad_q, 0.0)

        return x


class TransCP(nn.Module):
    """
    TransCP 主模型（带 SAM3 prior）

    Stage1 (coarse box): b0
    SAM3 prior: prompt_box -> mask_prob -> mask_bias
    Stage2 (refine): b1
    """

    def __init__(self, pretrained_weights, args=None):
        super().__init__()
        self.args = args

        # 1) Image encoder
        self.backbone = build_backbone(args)
        self.trans_encoder = build_visual_encoder(args)

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels,
            self.trans_encoder.d_model,
            kernel_size=1
        )

        # 2) Language encoder
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.bert_proj = nn.Linear(args.bert_output_dim, args.hidden_dim)
        self.bert_output_layers = args.bert_output_layers
        self.textmodel = build_LSTMBert(args, bert_model=self.bert)

        # 3) bbox regression (stage1 + stage2)
        self.bbox_regression = build_bbox_regression(args)
        self.bbox_regression_refine = build_bbox_regression(args)

        hidden_dim = self.trans_encoder.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # 4) Phrase embedding proj：768 -> hidden_dim
        self.lstm_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        # 4.5) fusion 配置
        self.use_fusion = bool(getattr(args, "use_fusion", True))
        self.fusion_use_stage1 = bool(getattr(args, "fusion_use_stage1", False))
        self.fusion_use_stage2 = bool(getattr(args, "fusion_use_stage2", True))

        fusion_heads = int(getattr(args, "fusion_nheads", 8))
        fusion_dropout = float(getattr(args, "fusion_dropout", 0.1))
        fusion_ffn_dim = getattr(args, "fusion_ffn_dim", None)
        fusion_gate_init = float(getattr(args, "fusion_gate_init", -4.0))

        if hidden_dim % fusion_heads != 0:
            raise ValueError(
                f"[Fusion] hidden_dim({hidden_dim}) must be divisible by fusion_nheads({fusion_heads})."
            )

        self.fusion = WordGuidedFusion(
            dim=hidden_dim,
            nheads=fusion_heads,
            dropout=fusion_dropout,
            ffn_dim=fusion_ffn_dim,
            gate_init=fusion_gate_init,
        )

        # 5) Debug / SAM3 prompt curriculum
        self.debug_sam3 = bool(getattr(args, "debug_sam3", False))
        self.debug_sam3_every = int(getattr(args, "debug_sam3_every", 50))

        self.sam3_prompt_use_gt_when_training = bool(getattr(args, "sam3_prompt_use_gt_when_training", True))
        self.sam3_prompt_warmup_epochs = int(getattr(args, "sam3_prompt_warmup_epochs", 5))
        self.sam3_prompt_mix_epochs = int(getattr(args, "sam3_prompt_mix_epochs", 10))

        # 6) SAM3 prior
        self.use_sam3 = bool(getattr(args, "use_sam3", True))
        if self.use_sam3:
            sam3_ckpt = getattr(args, "sam3_ckpt", None)
            if sam3_ckpt is None or str(sam3_ckpt).strip() == "":
                raise ValueError("args.sam3_ckpt is required when use_sam3=True")

            sam3_bpe = getattr(args, "sam3_bpe_path", None)
            sam3_res = int(getattr(args, "sam3_resolution", 1008))
            sam3_th = float(getattr(args, "sam3_confidence_threshold", 0.8))
            sam3_coord = str(getattr(args, "sam3_prompt_coord", "norm"))

            self.sam3 = Sam3BoxMaskTeacher(
                ckpt_path=sam3_ckpt,
                device=torch.device(args.device),
                bpe_path=sam3_bpe,
                resolution=sam3_res,
                confidence_threshold=sam3_th,
                prompt_coord=sam3_coord,
                autocast_dtype=torch.bfloat16,
                cache_text_out=True,
            )

            self.mask_adapter = MaskPriorAdapter(gamma_init=1.0, learnable_gamma=True)

        # 7) load pretrained (DETR)
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = {}
            for k in module_keys:
                prefix_k = prefix + '.' + k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)
        print("Weights of DETR are pre-loaded.")

    def forward(
        self,
        image, image_mask,
        word_id, word_mask,
        reason_id_0, reason_mask_0,
        reason_id_1, reason_mask_1,
        reason_id_2, reason_mask_2,
        gt_bbox: torch.Tensor = None,      # (B,4) cxcywh norm
        epoch: int = 0,
        step: int = 0,
    ):
        def _is_rank0():
            return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0

        def _dbg(msg: str):
            if self.debug_sam3 and _is_rank0() and (step % self.debug_sam3_every == 0):
                print(msg)

        def _box_stats_cxcywh(b):
            cx, cy, bw, bh = b.unbind(-1)
            return {
                "cx": (cx.mean().item(), cx.std().item(), cx.min().item(), cx.max().item()),
                "cy": (cy.mean().item(), cy.std().item(), cy.min().item(), cy.max().item()),
                "w":  (bw.mean().item(), bw.std().item(), bw.min().item(), bw.max().item()),
                "h":  (bh.mean().item(), bh.std().item(), bh.min().item(), bh.max().item()),
            }

        def _flat_hw_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if m is None:
                return None
            if m.dim() == 3:
                return m.flatten(1)
            return m

        # 1) Image features
        features, pos = self.backbone(NestedTensor(image, image_mask))
        src, mask = features[-1].decompose()
        bs, _, h_src, w_src = src.size()
        assert mask is not None

        img_feat, mask2d, pos_embed = self.trans_encoder(self.input_proj(src), mask, pos[-1])

        if mask2d is not None and mask2d.dim() == 3:
            h, w = int(mask2d.shape[-2]), int(mask2d.shape[-1])
        else:
            h, w = int(h_src), int(w_src)

        img_pad_mask = _flat_hw_mask(mask2d) if mask2d is not None else _flat_hw_mask(mask)
        token_mask_for_decoder = img_pad_mask

        # 2) Text features
        reason_id = torch.cat([reason_id_0, reason_id_1, reason_id_2], dim=1)
        reason_mask = torch.cat([reason_mask_0, reason_mask_1, reason_mask_2], dim=1)

        word_id_all = torch.cat([word_id, reason_id], dim=1)
        word_mask_all = torch.cat([word_mask, reason_mask], dim=1)

        disentangled_lang, bert_fea = self.textmodel(NestedTensor(word_id_all, word_mask_all))
        word_feat, word_mask2 = bert_fea.decompose()

        word_feat = self.bert_proj(word_feat)      # (B,L,C)
        word_feat = word_feat.permute(1, 0, 2)     # (L,B,C)

        assert word_mask2 is not None
        word_mask2 = word_mask2.flatten(1)         # (B,L) True=pad

        proj = self.lstm_proj(disentangled_lang)   # (B,C)
        projected_disentangled_lang = proj[:, :, None, None]

        # 2.5) Fusion
        guided_word_feat = word_feat
        if self.use_fusion and (self.fusion_use_stage1 or self.fusion_use_stage2):
            guided_word_feat = self.fusion(
                img_tokens=img_feat,
                word_tokens=word_feat,
                word_pad_mask=word_mask2,
                img_pad_mask=img_pad_mask,
            )

        # 3) Stage-1
        word_for_stage1 = guided_word_feat if (self.use_fusion and self.fusion_use_stage1) else word_feat
        hs0 = self.bbox_regression(
            img_feat,
            token_mask_for_decoder,
            pos_embed,
            word_for_stage1,
            word_mask2,
            projected_disentangled_lang,
            h, w,
            mask_bias=None,
        )

        b0 = self.bbox_embed(hs0).sigmoid()  # (B,4)
        out0 = {'pred_boxes': b0.unsqueeze(1)}

        _dbg(f"[DBG] epoch={epoch} step={step} bs={bs} feat_h,w={h},{w}")
        _dbg(f"[DBG] b0 stats: {_box_stats_cxcywh(b0)}")

        # 4) SAM3 prior
        mask_bias = None
        mask_prob = None
        score = None
        valid = None
        reliability = None

        if self.use_sam3:
            # ✅ 强制 SAM prompt 用 float32（避免 AMP 下 Half/Float index put 报错，也更适合喂给 SAM）
            prompt_box = b0.detach().float()

            if self.training and self.sam3_prompt_use_gt_when_training and (gt_bbox is not None):
                warm = self.sam3_prompt_warmup_epochs
                mix = self.sam3_prompt_mix_epochs

                # ✅ gt_bbox 对齐到 prompt_box 的 device/dtype（float32）
                gt_bbox_ = gt_bbox.detach().to(device=prompt_box.device, dtype=prompt_box.dtype)
                if gt_bbox_.dim() == 3 and gt_bbox_.shape[1] == 1:
                    gt_bbox_ = gt_bbox_.squeeze(1)

                if epoch < warm:
                    prompt_box = gt_bbox_
                    _dbg("[DBG] SAM prompt = GT (warmup)")
                elif mix > 0 and epoch < warm + mix:
                    t = float(epoch - warm) / float(mix)
                    p_gt = max(0.0, 1.0 - t)
                    use_gt = (torch.rand((bs,), device=prompt_box.device) < p_gt)

                    prompt_box = prompt_box.clone()
                    # ✅ dtype/device 已一致，不会再 Half/Float 报错
                    prompt_box[use_gt] = gt_bbox_[use_gt]
                    _dbg(f"[DBG] SAM prompt = MIX, p_gt={p_gt:.3f}, use_gt_ratio={use_gt.float().mean().item():.3f}")
                else:
                    _dbg("[DBG] SAM prompt = PRED (after warmup/mix)")

            with torch.no_grad():
                mask_prob = self.sam3(
                    image,
                    prompt_box,
                    out_size=(image.shape[-2], image.shape[-1])
                )

                if hasattr(self.sam3, "last_best_scores") and len(self.sam3.last_best_scores) == bs:
                    score = torch.tensor(self.sam3.last_best_scores, device=image.device, dtype=torch.float32).view(bs, 1).clamp(0.0, 1.0)
                else:
                    score = torch.ones((bs, 1), device=image.device, dtype=torch.float32)

                if hasattr(self.sam3, "last_valids") and len(self.sam3.last_valids) == bs:
                    valid = torch.tensor(self.sam3.last_valids, device=image.device, dtype=torch.float32).view(bs, 1)
                else:
                    valid = torch.ones((bs, 1), device=image.device, dtype=torch.float32)

                reliability = score * valid

            prior_out = self.mask_adapter(mask_prob, hw=(h, w), reliability=reliability)
            mask_bias = prior_out.mask_bias

            if mask_prob is not None:
                cov = (mask_prob > 0.5).float().mean().item()
                _dbg(f"[DBG] sam3 mask coverage@0.5 = {cov:.4f}")

            if mask_bias is not None:
                _dbg(
                    f"[DBG] mask_bias stats: mean={mask_bias.mean().item():.4f} std={mask_bias.std().item():.4f} "
                    f"min={mask_bias.min().item():.4f} max={mask_bias.max().item():.4f} nan={torch.isnan(mask_bias).any().item()}"
                )

            if reliability is not None:
                _dbg(
                    f"[DBG] reliability: mean={reliability.mean().item():.4f} "
                    f"min={reliability.min().item():.4f} max={reliability.max().item():.4f}"
                )

        # 5) Stage-2
        word_for_stage2 = guided_word_feat if (self.use_fusion and self.fusion_use_stage2) else word_feat
        hs1 = self.bbox_regression_refine(
            img_feat,
            token_mask_for_decoder,
            pos_embed,
            word_for_stage2,
            word_mask2,
            projected_disentangled_lang,
            h, w,
            mask_bias=mask_bias,
        )

        b1 = self.bbox_embed(hs1).sigmoid()

        _dbg(f"[DBG] b1 stats: {_box_stats_cxcywh(b1)}")
        delta = (b0 - b1).abs()
        _dbg(f"[DBG] |b0-b1| mean={delta.mean().item():.6f} max={delta.max().item():.6f}")

        out = {
            'pred_boxes': b1.unsqueeze(1),
            'aux_outputs': [out0],
            'pred_boxes_stage1': b0.unsqueeze(1),
            'pred_boxes_stage2': b1.unsqueeze(1),
        }

        if mask_prob is not None:
            out['sam3_mask'] = mask_prob
        if score is not None:
            out['sam3_score'] = score.view(bs)
        if valid is not None:
            out['sam3_valid'] = valid.view(bs)
        if reliability is not None:
            out['sam3_reliability'] = reliability.view(bs)

        return out


class VGCriterion(nn.Module):
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        super().__init__()
        self.weight_dict = weight_dict
        self.box_xyxy = box_xyxy
        self.loss_map = {'loss_boxes': self.loss_boxes}
        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4)
        target_boxes = target_boxes.reshape(-1, 4)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
            tgt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        else:
            src_boxes_xyxy = src_boxes
            tgt_boxes_xyxy = target_boxes

        loss_giou = 1 - box_ops.box_pair_giou(src_boxes_xyxy, tgt_boxes_xyxy)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos

        return losses

    def forward(self, outputs, targets):
        if isinstance(targets, dict):
            gt_boxes = targets['bbox']
        else:
            gt_boxes = targets.tensor_dict['bbox']

        pred_boxes = outputs['pred_boxes']
        B, Q, _ = pred_boxes.shape

        num_pos = avg_across_gpus(pred_boxes.new_tensor(B * Q))

        losses = {}
        losses.update(self.loss_loc(outputs, gt_boxes, num_pos))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                losses.update({k + f'_{i}': v for k, v in l_dict.items()})

        return losses


class PostProcess(nn.Module):
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        rsz_sizes, ratios, orig_sizes = target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes'].squeeze(1)

        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)

        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct

        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)

        boxes = boxes.clamp(min=0)

        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)

        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_vgmodel(args):
    device = torch.device(args.device)
    model = TransCP(pretrained_weights=args.load_weights_path, args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion.to(device)

    postprocessor = PostProcess(args.box_xyxy)
    return model, criterion, postprocessor
