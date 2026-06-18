# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

从 torch.nn.Transformer “拷贝+改造”的版本：
  1) positional encoding（pos_embed）不是直接加到输入上，而是在注意力里通过 with_pos_embed 加到 q/k
  2) encoder 最后“额外的 LN”按 DETR 论文习惯做了调整（这里由 normalize_before 控制）
  3) decoder 支持返回每一层的输出堆叠（return_intermediate_dec=True 时）
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    完整的 DETR Transformer（Encoder + Decoder）：
      - 输入 src: (B,C,H,W)
      - mask:     (B,H,W) True 表示 padding（不可见区域）
      - query_embed: (num_queries, C) 作为 decoder 的 query 位置嵌入
      - pos_embed:   (B,C,H,W) 视觉 token 的位置编码

    输出：
      - hs: decoder 每层输出 (num_layers, B, num_queries, C)（最后 transpose 处理）
      - memory: encoder 输出 (B,C,H,W)（把 token reshape 回 feature map）
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # -------------------------
        # Encoder：堆叠 N 层 TransformerEncoderLayer
        # -------------------------
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        # 如果 normalize_before=True，则 encoder 最后还会再做一次 LN（对应 PreNorm 体系）
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # -------------------------
        # Decoder：堆叠 N 层 TransformerDecoderLayer
        # -------------------------
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        # 初始化参数（线性层/注意力权重用 xavier）
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """对所有参数做 Xavier 初始化（只对 dim>1 的权重矩阵做）。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        src: (B,C,H,W)
        pos_embed: (B,C,H,W)
        query_embed: (num_queries,C)
        mask: (B,H,W) True=padding

        关键步骤：把 2D feature map 展平为 token 序列：
          (B,C,H,W) -> (HW,B,C)
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        # src tokens: (HW,B,C)
        src = src.flatten(2).permute(2, 0, 1)

        # pos tokens: (HW,B,C)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # query tokens: (num_queries,B,C)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # padding mask: (B,HW) True=padding
        mask = mask.flatten(1)

        # tgt 初始化为 0（decoder 的内容向量），真正提供“查询”信息的是 query_pos=query_embed
        tgt = torch.zeros_like(query_embed)

        # encoder 输出 memory: (HW,B,C)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # decoder 输出 hs:
        #   - 若 return_intermediate=True： (num_layers, num_queries, B, C)
        #   - 否则： (1, num_queries, B, C)
        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,          # encoder token 位置编码（给 cross-attn 的 key）
            query_pos=query_embed   # decoder query 位置编码（给 self-attn/cross-attn 的 query）
        )

        # hs: (num_layers, num_queries, B, C) -> (num_layers, B, num_queries, C)
        # memory: (HW,B,C) -> (B,C,HW) -> (B,C,H,W)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """
    Encoder：重复 N 层 TransformerEncoderLayer
    可选最终 norm（取决于 normalize_before 配置）
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,                  # attn_mask（很少用，一般为 None）
                src_key_padding_mask: Optional[Tensor] = None,  # (B,HW) True=padding
                pos: Optional[Tensor] = None):                  # (HW,B,C)
        output = src

        # 逐层 encoder
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos
            )

        # 若开启 pre_norm 体系，会在 encoder 最后再做一次 LN
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Decoder：重复 N 层 TransformerDecoderLayer
    如果 return_intermediate=True，会收集每一层输出（做 norm 后存下来），最终 stack 成 (N, ...)
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,                 # self-attn mask（一般 None）
                memory_mask: Optional[Tensor] = None,              # cross-attn mask（一般 None）
                tgt_key_padding_mask: Optional[Tensor] = None,     # decoder padding mask（DETR 中常 None）
                memory_key_padding_mask: Optional[Tensor] = None,  # encoder padding mask (B,HW)
                pos: Optional[Tensor] = None,                      # encoder pos (HW,B,C)
                query_pos: Optional[Tensor] = None):               # query pos (num_queries,B,C)
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )

            # 若需要返回每层输出：存下“norm 后”的结果
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 最后再做一次 norm（和 DETR 原实现一致）
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                # 上面已经 append 过最后一层的 norm(output)，这里替换成最终版
                intermediate.pop()
                intermediate.append(output)

        # 返回所有层：shape = (num_layers, num_queries, B, C)
        if self.return_intermediate:
            return torch.stack(intermediate)

        # 不返回中间层：人为扩一维，使得输出也符合 (1, ...)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    EncoderLayer：
      - self-attention（Q=K=src+pos, V=src）
      - FFN（两层线性 + activation）
      - 残差 + LayerNorm
    支持两种结构：
      - forward_post：Post-Norm（先 attn->add->norm；再 ffn->add->norm）
      - forward_pre ：Pre-Norm（先 norm 再做 attn/ffn，最后 add）
    由 normalize_before 控制
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self-attn：输入输出都是 (L,B,C)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN：Linear(d_model->ff)->act->drop->Linear(ff->d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 两个 LN，对应 attn block / ffn block
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # residual dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """把位置编码加到 token 上（DETR 的做法：只加到 Q/K）。"""
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,                 # attn_mask（一般 None）
                     src_key_padding_mask: Optional[Tensor] = None,     # (B,L) True=pad
                     pos: Optional[Tensor] = None):                     # (L,B,C)
        # Q=K=src+pos，V=src（pos 不加到 V）
        q = k = self.with_pos_embed(src, pos)

        # self-attention 输出 (L,B,C)；[0] 取 attn_output
        src2 = self.self_attn(
            q, k, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # 残差 + dropout + LN
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN：Linear->act->drop->Linear
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        # 残差 + dropout + LN
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # Pre-Norm：先 LN
        src2 = self.norm1(src)

        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        # 残差
        src = src + self.dropout1(src2)

        # 第二个 PreNorm + FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # 根据 normalize_before 选择 pre-norm 或 post-norm
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    DecoderLayer：
      1) self-attn（在 query/token 内部做注意力，Q=K=tgt+query_pos, V=tgt）
      2) cross-attn（Q=tgt+query_pos, K=memory+pos, V=memory）
      3) FFN
    同样支持 pre-norm / post-norm
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # decoder self-attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # cross-attn：query 来自 tgt，key/value 来自 memory
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 三个 LN：对应 self-attn / cross-attn / ffn 三个 block
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # residual dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """给 token 加上位置编码（decoder 中 query_pos 是可学习的 query embedding）。"""
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # 1) self-attn：Q=K=tgt+query_pos，V=tgt
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2) cross-attn：Q=tgt+query_pos，K=memory+pos，V=memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3) FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        # 1) pre-norm self-attn
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # 2) pre-norm cross-attn
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        # 3) pre-norm ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
            pos, query_pos
        )


def _get_clones(module, N):
    """深拷贝 N 份层，用于堆叠 encoder/decoder。"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """
    构建 DETR Transformer（encoder+decoder）：
      - return_intermediate_dec=True：decoder 返回每一层输出（DETR 默认用于每层预测）
    """
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,      # True=PreNorm，False=PostNorm
        return_intermediate_dec=True,
    )


class VisualEncoder(nn.Module):
    """
    只要 encoder 的版本（你 TransCP 里用这个）：
      输入 src: (B,C,H,W)
      输出 memory: (HW,B,C)
           mask:   (B,HW)
           pos:    (HW,B,C)
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Xavier 初始化。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """
        src: (B,C,H,W)
        pos_embed: (B,C,H,W)
        mask: (B,H,W) True=padding
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        # (B,C,H,W)->(HW,B,C)
        src = src.flatten(2).permute(2, 0, 1)

        # (B,C,H,W)->(HW,B,C)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # (B,H,W)->(B,HW)
        mask = mask.flatten(1)

        # encoder 输出 memory: (HW,B,C)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # 返回 memory 以及展开后的 mask/pos_embed（后续 bbox_regression 会用）
        return memory, mask, pos_embed


def build_visual_encoder(args):
    """构建只包含 encoder 的视觉编码器（TransCP 用它替代完整 DETR transformer）。"""
    return VisualEncoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm
    )


def _get_activation_fn(activation):
    """
    根据字符串返回激活函数：
      - relu/relu_inplace -> nn.ReLU(inplace=True)
      - gelu -> F.gelu
      - glu  -> F.glu
    """
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
