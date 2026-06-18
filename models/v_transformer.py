# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

从 torch.nn.Transformer 拷贝并做了小改动：
  - positional encoding（pos_embed）通过 with_pos_embed 加到注意力的 Q/K（而非直接加到输入）
  - encoder 末尾的额外 LN 是否存在由 normalize_before 决定（这里保留 norm 选项）
  - 你这份是“只保留 encoder 的版本”（VisionEncoder），没有 decoder
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class VisionEncoder(nn.Module):
    """
    一个“只包含 Transformer Encoder”的模块（常用于对 token 序列做上下文编码）。

    和 DETR 的完整 Transformer 不同：
      - 没有 decoder
      - forward 直接把输入 src 送进 encoder
      - 要求 src 已经是 token 形式（例如 L,B,C），而不是 (B,C,H,W)

    参数：
      d_model: token 维度（C）
      nhead: 多头注意力的 head 数
      num_encoder_layers: encoder 层数
      normalize_before: True=PreNorm，False=PostNorm
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # 构建单层 encoder layer（包含 self-attn + FFN）
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )

        # 只有当 normalize_before=True（PreNorm）时，encoder 最后会额外做一次 LayerNorm
        # normalize_before=False 时为 None（也就是不做最后总 LN）
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        # 把单层 clone N 份形成 encoder 堆叠
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 初始化参数（Xavier）
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """对权重矩阵做 Xavier 初始化（dim>1 才是矩阵）。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """
        src: (L, B, C)
            - L 是 token 序列长度（例如 HW 或 1+HW 等）
            - B batch size
            - C=d_model

        mask: (B, L)  (key_padding_mask)
            - True 表示 padding / 不可见 token
            - 会在 attention 里把对应位置 mask 掉（置 -inf）

        pos_embed: (L, B, C)
            - token 的位置编码
            - 通过 with_pos_embed 加到 Q/K 上

        返回：
          encoded: (L, B, C)
        """
        return self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)


class TransformerEncoder(nn.Module):
    """
    标准 Transformer Encoder：
      - layers: N 个 TransformerEncoderLayer
      - norm: 可选最终 LayerNorm
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 深拷贝 num_layers 份，形成堆叠
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,                  # attn_mask（一般不用）
                src_key_padding_mask: Optional[Tensor] = None,  # (B,L) True=padding
                pos: Optional[Tensor] = None):                  # (L,B,C)
        output = src

        # 逐层编码
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos
            )

        # 可选最终 LN（通常只在 pre_norm 体系下启用）
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 的一层：
      1) Multihead Self-Attention
      2) FFN（两层全连接）
      3) 残差 + LayerNorm

    支持两种结构：
      - forward_post: Post-Norm（DETR 默认常用）
      - forward_pre : Pre-Norm（更稳，深层训练更容易）

    关键点：pos_embed 只加到 Q/K（DETR 写法）
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self-attention：默认用 nn.MultiheadAttention（不带相对位置）
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 你注释里提到可替换为 MHAttentionRPE（相对位置编码版本）
        # self.self_attn = MHAttentionRPE(d_model, nhead, dropout=dropout)

        # FFN：Linear(d_model->ff)->act->drop->Linear(ff->d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 两个 LayerNorm（分别对应 attn block 和 ffn block）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 两个 residual dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数（relu/gelu/glu）
        self.activation = _get_activation_fn(activation)

        # True=PreNorm，False=PostNorm
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        把 pos_embed 加到 token 上：
          - DETR 的习惯：pos 只加到 attention 的 Q/K 上，不加到 V 上
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,              # attn_mask（一般 None）
                     src_key_padding_mask: Optional[Tensor] = None,  # (B,L) True=pad
                     pos: Optional[Tensor] = None):                  # (L,B,C)
        # Post-Norm 结构：
        # 1) self-attn -> add&norm
        # 2) ffn -> add&norm

        # Q=K=src+pos, V=src
        q = k = self.with_pos_embed(src, pos)

        # self-attn 输出 [0] 是 attn_output，形状 (L,B,C)
        src2 = self.self_attn(
            q, k, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # residual + dropout + LN
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # residual + dropout + LN
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # Pre-Norm 结构：
        # 1) norm -> self-attn -> add
        # 2) norm -> ffn -> add

        # 先 LN
        src2 = self.norm1(src)

        # Q=K=src2+pos, V=src2
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # residual
        src = src + self.dropout1(src2)

        # 第二个 LN + FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        # residual
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 根据 normalize_before 选择 pre/post 版本
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    """深拷贝 N 份，用于堆叠 encoder layers。"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_v_transformer(args):
    """
    构建一个 VisionEncoder（只 encoder）实例。
    你这里写死了一些超参：
      - d_model=256（与你 transcp_visual 里的 v_transformer 输入一致）
      - dropout=0.0（注意力/FFN 都不drop）
      - nhead=8
      - dim_feedforward=2048
      - num_encoder_layers=6
      - normalize_before=False -> PostNorm
    """
    return VisionEncoder(
        d_model=256,
        dropout=0.,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        normalize_before=False,
    )


def _get_activation_fn(activation):
    """
    根据字符串返回激活函数（这里返回的是函数对象，而不是 nn.Module）：
      - relu -> F.relu
      - gelu -> F.gelu
      - glu  -> F.glu
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
