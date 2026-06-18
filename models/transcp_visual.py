# models/transcp_visual.py
# -*- coding: utf-8 -*-

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import math

from .neck import PrototypeLearner          # 你自己的 prototype 模块（VQ/字典式视觉原型）
from .v_transformer import build_v_transformer  # 视觉 transformer（回归 token + visual tokens）


class BboxRegression(nn.Module):
    """
    BBoxRegression：视觉解码器（bbox_regression）的一部分

    输入：
      - img_feat: (HW, B, C)         视觉 token（通常来自 backbone+encoder）
      - img_key_padding_mask: (B, HW) True=padding/无效 token
      - pos_embed: (HW, B, C)        视觉位置编码
      - word_feat / word_mask / projected_disentangled_lang: 语言相关特征
      - h,w: token 网格大小（通常 20,20 -> HW=400）
      - mask_bias: (HW,B,1) 可选的 mask prior（来自 SAM3）

    输出：
      - hs[0]: (B, 256) 或 (B, C) 的回归 token 表示（供 bbox_head 去预测框）
    """
    def __init__(self, cfg):
        super().__init__()
        args = cfg.copy()                    # cfg 来自 args.model_config['decoder']
        layer_type = args.pop('type')        # 指定用哪个模块（如 VisualDenstanglingPrototype）
        self.layer = _MODULES[layer_type](**args)

        # 对 transformer 输出做 LayerNorm（稳定训练）
        self.norm = nn.LayerNorm(256)

        # 注意：你原仓库强依赖 20x20=400（很多地方默认 HW=400）
        self.num_visu_token = 576
        num_total = self.num_visu_token + 1  # +1 是回归 token（reg_token）

        # 构建用于回归的视觉 transformer（输入 reg_token + 多模态视觉 tokens）
        self.v_transformer = build_v_transformer(args)

        # 为 (1 + HW) 个 token 准备绝对位置 embedding（固定长度 401）
        self.v_pos_embed = nn.Embedding(num_total, 256)

        # 回归 token：类似 DETR 的 [CLS]，用它聚合信息用于输出 bbox 表示
        self.reg_token = nn.Embedding(1, 256)

        self._reset_parameters()

    def _reset_parameters(self):
        # 对所有二维以上参数做 Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        img_feat, img_key_padding_mask, pos_embed,
        word_feat, word_mask, projected_disentangled_lang, h, w,
        mask_bias: Optional[torch.Tensor] = None,  # (HW,B,1) SAM3 prior
    ):
        hw, bs, c = img_feat.shape  # hw=H*W, bs=batch, c=channel

        # -------------------------
        # 1) Visual Context Disentangling + Prototype Learning + Multi-modal fusion
        # -------------------------
        # self.layer 输出 x_multi_modal：形状通常为 (HW, B, 256)
        # 内部可能会融合语言信息、prototype、mask_bias 等
        x_multi_modal = self.layer(
            img_feat, img_key_padding_mask, pos_embed,
            word_feat, word_mask, projected_disentangled_lang, h, w,
            mask_bias=mask_bias,
        )

        # -------------------------
        # 2) 构造回归 token + 拼接视觉 token，送入 v_transformer
        # -------------------------
        # reg token: (1, B, 256)
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)

        # reg token 的 padding mask：全 False，表示有效
        tgt_mask = torch.zeros((bs, 1), device=tgt_src.device, dtype=torch.bool)

        # 拼接：v_src = (1+HW, B, 256)
        v_src = torch.cat([tgt_src, x_multi_modal], dim=0)

        # 拼接 mask：v_mask = (B, 1+HW)
        v_mask = torch.cat([tgt_mask, img_key_padding_mask], dim=1)

        # 位置 embedding：固定 (1+400=401, 256) -> 扩成 (401, B, 256)
        # ⚠️这里仍然按 401 固定（对应 20x20）
        v_pos = self.v_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # transformer 输出： (1+HW, B, 256)
        output = self.v_transformer(v_src, v_mask, v_pos)

        # LayerNorm 稳定输出
        hs = self.norm(output) if self.norm is not None else output

        # 只取第 0 个 token（reg token），作为 bbox 表示
        return hs[0]


class VisualDenstanglingPrototype(nn.Module):
    """
    VisualDenstanglingPrototype：
    - 先过若干 extra_encoder_layers（图像 query 去 attend 文本）
    - 得到 img_feat = concat([orig_img_feat, fuse_img_feat])  (dim=-1 上拼起来)
    - chunk(2) 拆分出 ori_img / dis_img
    - dis_img 送入 PrototypeLearner 得到 prototype-enhanced 的 dis_img_vd
    - 再与 projected_disentangled_lang 做融合（tanh 门控）
    - 最后（可选）注入 mask_bias（SAM3 prior）
    """
    def __init__(self, num_queries, query_dim,
                 return_intermediate=False,
                 extra_layer=None, num_extra_layers=1):
        super().__init__()

        # extra_layer 指定类型（例如 DiscriminativeFeatEncLayer）
        args = extra_layer.copy()
        layer_type = args.pop('type')

        # 构造一个 encoder layer，然后 clone N 份
        extra_encoder_layer = _MODULES[layer_type](**args)
        self.extra_encoder_layers = _get_clones(extra_encoder_layer, num_extra_layers)

        self.return_intermediate = return_intermediate

        # 视觉/文本 query embedding（这里其实没在 forward 用到，可能是原版遗留）
        self.vis_query_embed = nn.Embedding(num_queries, query_dim)
        self.text_query_embed = nn.Embedding(num_queries, query_dim)

        # prototype discovery module：视觉原型字典（你自定义模块）
        self.prototypelearner = PrototypeLearner(num_tokens=2048, decay=0.4)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 有 pos 就加上（标准 transformer 写法）
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        img_feat, img_key_padding_mask=None, pos=None,
        word_feat=None, word_key_padding_mask=None,
        projected_disentangled_lang=None, h=20, w=20,
        mask_bias: Optional[torch.Tensor] = None,  # (HW,B,1)
    ):
        hw, bs, c = img_feat.shape

        # -------------------------
        # 1) Visual Context Disentangling：图像 token attend 文本 token
        # -------------------------
        for layer in self.extra_encoder_layers:
            img_feat = layer(
                img_feat, img_key_padding_mask, pos,
                word_feat, word_key_padding_mask, None
            )

        # -------------------------
        # 2) split to ori / dis
        # 你这里 layer 输出是 cat([orig_img_feat, fuse_img_feat], dim=-1)
        # 所以 dim=-1 上一分为二
        # -------------------------
        img_feat_srcs = img_feat.chunk(2, dim=-1)
        dis_img = img_feat_srcs[1]   # “可区分特征”分支
        ori_img = img_feat_srcs[0]   # 原始分支（此处没有继续用）

        # -------------------------
        # 3) prototype embedding：给 dis_img 注入视觉原型
        # -------------------------
        prototype_out = self.prototypelearner(dis_img, h, w)

        # embedded_pt：按你 PrototypeLearner 的实现，最终应该是 (HW, B, C)
        dis_img_vd = prototype_out["embedded_pt"]

        # ---- 兜底：把 embedded_pt 统一成 (HW,B,C) ----
        # 你加的兜底很重要：因为不同实现可能返回 (B,C,H,W)
        if dis_img_vd.dim() == 4:
            # 情况1： (B,C,H,W) -> (HW,B,C)
            if dis_img_vd.shape[0] == bs:
                dis_img_vd = dis_img_vd.flatten(2).permute(2, 0, 1).contiguous()
            # 情况2：奇怪维度（例如 (HW,B,C,1)），做压缩处理
            elif dis_img_vd.shape[1] == bs:
                if dis_img_vd.shape[-1] == 1:
                    dis_img_vd = dis_img_vd[..., 0].contiguous()
                else:
                    dis_img_vd = dis_img_vd.mean(dim=-1).contiguous()
            else:
                raise RuntimeError(f"[embedded_pt] unexpected shape: {tuple(dis_img_vd.shape)}")
        elif dis_img_vd.dim() != 3:
            raise RuntimeError(f"[embedded_pt] unexpected dim: {dis_img_vd.dim()} shape={tuple(dis_img_vd.shape)}")

        # 统一后的 dis_img_vd: (HW,B,C)
        hw2, bs2, c2 = dis_img_vd.shape

        # hw 不一致：强行以 dis_img_vd 为准（理论上应一致）
        if hw2 != hw:
            hw = hw2

        # batch 必须一致
        if bs2 != bs:
            raise RuntimeError(f"[embedded_pt] batch mismatch: {bs2} vs {bs}")

        # ---- 兜底：h*w 不一致时推断 ----
        # 期望 HW=h*w；若不一致，尝试把 HW 当作正方形推断 side
        if h * w != hw:
            side = int(round(math.sqrt(hw)))
            if side * side == hw:
                h, w = side, side
            else:
                raise RuntimeError(f"[VisualDenstanglingPrototype] hw={hw} but h*w={h*w} not match and not square.")

        # ---- 统一 projected_disentangled_lang 成 (B,C,1,1) ----
        # projected_disentangled_lang：语言侧投影出来的向量（用于门控融合）
        lang = projected_disentangled_lang
        if lang is None:
            # 没给就用全 1（相当于不改变视觉）
            lang = dis_img_vd.new_ones((bs, c2, 1, 1))
        else:
            # 常见三种输入：
            #   (B,C)   -> (B,C,1,1)
            #   (B,C,1) -> (B,C,1,1)
            #   (B,C,H,W) -> 直接用
            if lang.dim() == 2:
                lang = lang[:, :, None, None]
            elif lang.dim() == 3:
                lang = lang[:, :, :, None]
            elif lang.dim() == 4:
                pass
            else:
                # 更奇怪的 shape，强行 reshape 成 (B,?,1,1)
                lang = lang.view(bs, -1, 1, 1)

        # -------------------------
        # 4) 关键：稳定的 4D map 融合
        # -------------------------
        # dis_img_vd: (HW,B,C) -> dis_map: (B,C,H,W)
        dis_map = dis_img_vd.permute(1, 2, 0).contiguous().view(bs, c2, h, w)

        # tanh 门控融合：视觉与语言都经过 tanh 压到 [-1,1]
        # 这样做的效果：抑制极端值、稳定训练
        x_map = torch.tanh(dis_map) * torch.tanh(lang)   # (B,C,H,W)

        # 再回到 token 形式：(B,C,H,W) -> (HW,B,C)
        x_multi_modal = x_map.flatten(2).permute(2, 0, 1).contiguous()

        # -------------------------
        # 5) ✅ 注入 SAM3 mask prior
        # -------------------------
        # mask_bias: (HW,B,1)，通常来自 mask_prob 下采样后的 bias
        # 这里用乘法 (1 + bias) 做缩放：
        #   bias>0 强化该位置 token
        #   bias=0 不变
        #   bias<0（理论上 mask_prob 是 [0,1]，通常不会<0）
        if mask_bias is not None:
            x_multi_modal = x_multi_modal * (1.0 + mask_bias)

        return x_multi_modal


class DiscriminativeFeatEncLayer(nn.Module):
    """
    DiscriminativeFeatEncLayer：
    - img_query attend text：得到 shared semantic F_s
    - 分别把 F_s 和 img_feat 投影到同维度空间
    - 计算余弦相似度作为 dis_coef（可区分系数）
    - 用 exp(- (1-cos)^pow / (2*sigma^2)) 做 soft gating（高相似->权重大）
    - 输出 cat([orig_img_feat, fuse_img_feat]) 供后续 chunk(2)
    """
    def __init__(self, d_model, img2text_attn_args=None, img_query_with_pos=True,
                 discrimination_coef_settings=None):
        super().__init__()
        args = img2text_attn_args.copy()
        # 多头注意力：可以是标准 MHA 或带 RPE 的 MHAttentionRPE
        self.img2text_attn = MULTIHEAD_ATTNS[args.pop('type')](**args)
        self.img_query_with_pos = img_query_with_pos

        # 两个 MLP：将共享语义与图像特征投影到可比较空间
        self.text_proj = MLP(**discrimination_coef_settings['text_proj'])
        self.img_proj = MLP(**discrimination_coef_settings['img_proj'])

        # gating 的超参：pow/scale/sigma
        self.tf_pow = discrimination_coef_settings.get('pow')
        self.tf_scale = Parameter(torch.Tensor([discrimination_coef_settings.get('scale')]))
        self.tf_sigma = Parameter(torch.Tensor([discrimination_coef_settings.get('sigma')]))

        self.norm_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat  # 保留原始图像 token

        # query 是否加 pos（常见做法：query=img+pos）
        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat

        # -------------------------
        # 1) 图像 query attend 文本 key/value，得到共享语义 F_s
        # -------------------------
        # self.img2text_attn 返回 (attn_output, attn_weights)
        # F_s: (HW, B, C)
        F_s = self.img2text_attn(
            query=img_query,
            key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat,
            key_padding_mask=word_key_padding_mask
        )[0]

        # -------------------------
        # 2) 投影并计算可区分系数 dis_coef
        # -------------------------
        # text_embed/img_embed：投影到某个维度（由 MLP 配置决定）
        text_embed = self.text_proj(F_s)
        img_embed = self.img_proj(img_feat)

        # 余弦相似度：normalize 后点积
        # dis_coef: (HW, B, 1)
        dis_coef = (F.normalize(img_embed, p=2, dim=-1) *
                    F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)

        # 用高斯形态的函数把相似度映射为正系数（越相似越大）
        # dis_coef = scale * exp(- (1-cos)^pow / (2*sigma^2))
        dis_coef = self.tf_scale * torch.exp(
            - (1 - dis_coef).pow(self.tf_pow) / (2 * self.tf_sigma ** 2)
        )

        # -------------------------
        # 3) 用 dis_coef 对图像特征做 gating（再 concat 原始+fuse）
        # -------------------------
        fuse_img_feat = self.norm_img(img_feat) * dis_coef

        # 输出维度翻倍：cat([orig, fuse]) -> (HW, B, 2C)
        # 后面 VisualDenstanglingPrototype 会 chunk(2) 拆开
        return torch.cat([orig_img_feat, fuse_img_feat], dim=-1)


# decoder/encoder 里可选模块注册表
_MODULES = {
    'VisualDenstanglingPrototype': VisualDenstanglingPrototype,
    'DiscriminativeFeatEncLayer': DiscriminativeFeatEncLayer,
}


def build_bbox_regression(args):
    # 从 args.model_config['decoder'] 构建 bbox regression 模块
    return BboxRegression(args.model_config['decoder'])


def _get_clones(module, N):
    # 深拷贝 N 份 module（每层独立参数）
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MHAttentionRPE(nn.Module):
    """
    Multi-Head Attention with Relative Position Embedding (RPE)

    核心思想：
      标准 attention = q*k
      加上相对位置偏置项：q*k_pos(rel)
      这里对 x/y 两个方向分别做相对位置 embedding，再相加

    输入：
      query: (tgt_len, B, C)   这里 tgt_len 通常是 HW
      key  : (src_len, B, C)   这里 src_len 通常也是 HW
      value: (src_len, B, C)
      key_padding_mask: (B, HW) True=padding

    输出：
      attn_output: (tgt_len, B, C)
      attn_weights 或 raw attention（按 return_raw_attention 控制）
    """

    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-24, 24], pos_y_range=[-24, 24], pos_index_offset=24,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h                      # 每个 head 的维度
        self.h = h                                   # head 数
        self.scaling = float(self.d_k) ** -0.5       # 缩放因子 1/sqrt(d_k)
        self.return_raw_attention = return_raw_attention

        # in_proj：把输入一次性映射到 Q/K/V（与 nn.MultiheadAttention 类似）
        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None                             # 可保存最后一次 attention（调试用）
        self.dropout_p = dropout
        self._reset_parameters()

        # RPE：可学习 embedding 或固定 sine embedding
        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            # learnable：x/y 相对位置的 embedding（各占一半维度）
            self.pos_x = nn.Embedding(pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
            self.pos_y = nn.Embedding(pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            # fixed sine：返回 (range, C/2)
            pos_x, pos_y = position_embedding_sine(
                d_model // 2, normalize=True, x_range=pos_x_range, y_range=pos_y_range
            )
            self.register_buffer('pos_x', pos_x)  # [x_range_len, C/2]
            self.register_buffer('pos_y', pos_y)  # [y_range_len, C/2]

        # 把相对坐标偏移到非负索引（例如 [-20,20] + 20 -> [0,40]）
        self.pos_index_offset = pos_index_offset

    def _reset_parameters(self):
        # 初始化 QKV projection 与 out projection
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bs, dim = query.size()
        src_len, _, _ = key.size()

        # -------------------------
        # 1) 计算 Q/K/V
        # -------------------------
        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim * 2], self.in_proj_bias[dim:dim * 2]
        weight_v, bias_v = self.in_proj_weight[dim * 2:], self.in_proj_bias[dim * 2:]

        # (tgt_len,B,C) x (C,C) -> (tgt_len,B,C)
        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        # reshape 成多头形式：
        # q: (B*h, tgt_len, d_k)
        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)

        # k: (B*h, d_k, src_len) 方便 bmm
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)

        # v: (B*h, src_len, d_k)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        # scale
        q = q * self.scaling

        # 标准 attention logits： (B*h,tgt_len,d) x (B*h,d,src_len) -> (B*h,tgt_len,src_len)
        attn_weights = torch.bmm(q, k)

        # -------------------------
        # 2) 计算相对位置（依赖 key_padding_mask 推断网格坐标）
        # -------------------------
        if key_padding_mask is None:
            raise ValueError("MHAttentionRPE requires key_padding_mask for relative positions.")

        bs2, HW = key_padding_mask.size()
        assert bs2 == bs, "key_padding_mask batch mismatch"
        assert HW == tgt_len, "key_padding_mask length mismatch with query"

        # 动态推断 H,W（必须是平方格：例如 20*20=400）
        side = int(round(math.sqrt(HW)))
        if side * side != HW:
            raise ValueError(f"MHAttentionRPE expects square HW, got HW={HW}")
        H = W = side

        # img_mask: (B,H,W) True=有效位置（因为 key_padding_mask True=padding，所以取反）
        img_mask = ~key_padding_mask.view(bs, H, W)

        # 这里用 cumsum 得到每个位置的“坐标编号”（从 1 开始）
        # yy/xx 最终展平到 (B,HW)
        yy = img_mask.cumsum(1, dtype=torch.float32).view(bs, -1)  # y 坐标 1~H
        xx = img_mask.cumsum(2, dtype=torch.float32).view(bs, -1)  # x 坐标 1~W

        # 两两差值：得到相对位置 diff
        # diff_yy/diff_xx: (B,HW,HW)
        diff_yy = yy[:, :, None] - yy[:, None, :]
        diff_xx = xx[:, :, None] - xx[:, None, :]

        # -------------------------
        # 3) 将相对位置 embedding 投影到 K 空间，计算 q * k_pos(rel)
        # -------------------------
        if self.learnable_pos_embed:
            # (range_len, C/2) x (C/2, C) -> (range_len, C)
            k_posy = self.pos_y.weight.matmul(weight_k.t()[:dim // 2])
            k_posx = self.pos_x.weight.matmul(weight_k.t()[dim // 2:])
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim // 2])
            k_posx = self.pos_x.matmul(weight_k.t()[dim // 2:])

        # reshape 成多头并扩到 batch：
        # k_posy: (B*h, d_k, y_range_len)
        k_posy = k_posy.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1).reshape(
            -1, bs * self.h, dim // self.h
        ).permute(1, 2, 0)

        # k_posx: (B*h, d_k, x_range_len)
        k_posx = k_posx.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1).reshape(
            -1, bs * self.h, dim // self.h
        ).permute(1, 2, 0)

        # q: (B*h, HW, d_k)
        # bmm 得到对每个 token 与每个相对位置 index 的 attention：
        # posy_attn_weights: (B, h, HW, y_range_len)
        posy_attn_weights = torch.bmm(q, k_posy).view(bs, self.h, HW, -1)
        posx_attn_weights = torch.bmm(q, k_posx).view(bs, self.h, HW, -1)

        # 把 diff 转成索引（+offset -> 非负）
        # diff_yy_idx/diff_xx_idx: (B,h,HW,HW)
        diff_yy_idx = diff_yy[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset

        # gather：为每对 token(i,j) 取对应的相对位置偏置
        posy_attn_weights = torch.gather(posy_attn_weights, -1, diff_yy_idx.long())
        posx_attn_weights = torch.gather(posx_attn_weights, -1, diff_xx_idx.long())

        # 合并 x/y 偏置，展平回 (B*h, HW, HW)
        pos_attn_weights = (posy_attn_weights + posx_attn_weights).view(bs * self.h, HW, -1)

        # 把相对位置偏置加到原始 attn logits 上
        attn_weights = attn_weights + pos_attn_weights

        # -------------------------
        # 4) padding mask：把 padding 位置置为 -inf
        # -------------------------
        attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,HW)
            float('-inf')
        )
        attn_weights = attn_weights.view(-1, tgt_len, src_len)

        raw_attn_weights = attn_weights  # 保存未 softmax 的 logits（可选返回）

        # softmax 得到概率 attention
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # -------------------------
        # 5) attention * V 得到输出
        # -------------------------
        # (B*h,HW,HW) x (B*h,HW,d_k) -> (B*h,HW,d_k)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        # 合并 head：(HW,B,C)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)

        # out projection
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # 需要 raw attention 就返回 logits，否则返回 softmax 后
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


# attention 模块注册表
MULTIHEAD_ATTNS = {
    'MultiheadAttention': nn.MultiheadAttention,
    'MHAttentionRPE': MHAttentionRPE,
}


class MLP(nn.Module):
    """Very simple multi-layer perceptron (FFN)

    input_dim -> hidden_dim*(num_layers-1) -> output_dim
    最后一层不加 ReLU
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip(
                [input_dim] + h, h + [output_dim]
            ))
        else:
            self.layers = nn.ModuleList([])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 前 num_layers-1 层 ReLU，最后一层线性输出
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def position_embedding_sine(
    num_pos_feats=64,
    temperature=10000,
    normalize=False,
    scale=None,
    x_range=[-24, 24],
    y_range=[-24, 24],
    device=None
):
    """
    生成一维坐标的 sine-cosine 位置编码（用于相对位置 embedding 查表）

    输入：
      - x_range/y_range：例如 [-20,20] 表示相对位移范围
      - num_pos_feats：一半维度（最终输出维度会是 2*num_pos_feats）

    输出：
      - pos_x: (len(x_range), 2*num_pos_feats)
      - pos_y: (len(y_range), 2*num_pos_feats)
    """
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    # 相对坐标取值：[-20,...,20]
    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device)
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)

    if normalize:
        eps = 1e-6
        # 注意：这里除以最后一个元素（最大值）做归一化，再乘 scale
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    # 频率项：temperature^(2i/D)
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    # (range_len, num_pos_feats)
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t

    # 交替 sin/cos 并拼接 -> (range_len, 2*num_pos_feats)
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y
