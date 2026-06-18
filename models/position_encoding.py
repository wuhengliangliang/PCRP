# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved  # 版权说明（DETR 系列常见）
"""
Various positional encodings for the transformer.
# 本文件：为 Transformer 提供位置编码（positional encoding）
# - Sine/Cos（固定、可泛化到任意尺寸）
# - Learned（绝对位置、可学习 embedding）
"""
import math  # 用于 2*pi 等
import torch  # PyTorch
from torch import nn  # nn.Module / nn.Embedding 等

from util.misc import NestedTensor  # NestedTensor: (tensors, mask) 打包结构


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    # 说明：
    # - 原始 Transformer 是 1D 序列，用正弦/余弦编码序列位置
    # - 这里把它扩展到 2D 图像：分别对 x/y 位置做 sin/cos，再拼接
    # - 还支持 mask：对 padding 区域不算位置（保证变长输入可用）
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()  # 初始化父类
        self.num_pos_feats = num_pos_feats  # 每个方向（x 或 y）使用的特征维度数（最终会拼成 2*num_pos_feats）
        self.temperature = temperature      # 频率基数（经典默认 10000）
        self.normalize = normalize          # 是否把坐标归一化到 [0, 2*pi]
        if scale is not None and normalize is False:
            # 如果给了 scale，但 normalize=False，会让 scale 无意义，因此强制要求 normalize=True
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi  # 默认将归一化坐标映射到 [0, 2π]（经典做法）
        self.scale = scale  # 保存 scale

    def forward(self, tensor_list: NestedTensor):
        # tensor_list.tensors: (B, C, H, W)  # 输入特征图（或图像）
        x = tensor_list.tensors

        # tensor_list.mask: (B, H, W)  # True/False 表示哪些位置是 padding（依你实现，通常 True=pad）
        mask = tensor_list.mask
        assert mask is not None  # 必须有 mask 才能正确构建位置（DETR 里一直有）

        # not_mask: (B,H,W)  # 非 padding 区域为 True（可累加计数）
        not_mask = ~mask

        # y_embed: (B,H,W)  # 沿着高度维（dim=1）做累加：得到每个像素的“y坐标序号”
        # cumsum 在 not_mask 上做，相当于：padding 区域不增加坐标
        y_embed = not_mask.cumsum(1, dtype=torch.float32)

        # x_embed: (B,H,W)  # 沿着宽度维（dim=2）做累加：得到“x坐标序号”
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # 如果 normalize=True，把坐标归一化到固定尺度（更利于不同分辨率泛化）
        if self.normalize:
            eps = 1e-6  # 防止除零
            # y_embed[:, -1:, :] 是每行最后一个有效坐标值（按 batch、按列）
            # y 归一化后再乘 self.scale（默认 2*pi）
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # x_embed[:, :, -1:] 是每列最后一个有效坐标值（按 batch、按行）
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t: (num_pos_feats,)  # 生成频率分母（不同维度用不同尺度）
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        # 经典公式：temperature^(2i/num_pos_feats)，i 为维度索引的一半（因为 sin/cos 成对）
        # dim_t // 2：让 (0,1) 同频率，(2,3) 同频率...
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # pos_x: (B,H,W,num_pos_feats)  # 把 x 坐标扩展一个维度，然后除以 dim_t 得到不同频率的相位
        pos_x = x_embed[:, :, :, None] / dim_t

        # pos_y: (B,H,W,num_pos_feats)  # 同理对 y
        pos_y = y_embed[:, :, :, None] / dim_t

        # 对偶维度做 sin/cos 交错：
        # pos_x[..., 0::2] 用 sin，pos_x[..., 1::2] 用 cos
        # stack 后 flatten(3)：把 (sin,cos) 合并回最后一维 -> (B,H,W,num_pos_feats)
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        # y 同理
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        # 拼接 (y,x)：(B,H,W,2*num_pos_feats)
        # 然后 permute 到 (B, 2*num_pos_feats, H, W)（符合 CNN/Transformer 中常用通道优先格式）
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos  # 返回位置编码，与输入特征图在空间维度 (H,W) 对齐


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    # 可学习绝对位置编码：
    # - 为每个行号/列号学习一个 embedding
    # - 然后拼接得到每个 (y,x) 位置的 pos 向量
    # 注意：这里 row/col embedding 的最大长度固定为 50
    # 如果特征图 H/W > 50 会越界（需要改 embedding 大小）
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()  # 初始化父类
        self.row_embed = nn.Embedding(50, num_pos_feats)  # 行位置 embedding：最多 50 行
        self.col_embed = nn.Embedding(50, num_pos_feats)  # 列位置 embedding：最多 50 列
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        # 用均匀分布初始化 embedding 权重
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors  # (B,C,H,W)
        h, w = x.shape[-2:]      # 取出 H,W

        # i: (w,) 列索引 [0..w-1]
        i = torch.arange(w, device=x.device)

        # j: (h,) 行索引 [0..h-1]
        j = torch.arange(h, device=x.device)

        # x_emb: (w,num_pos_feats)  # 每个列位置的 embedding
        x_emb = self.col_embed(i)

        # y_emb: (h,num_pos_feats)  # 每个行位置的 embedding
        y_emb = self.row_embed(j)

        # 构建每个 (y,x) 位置的 pos：
        # x_emb.unsqueeze(0): (1,w,d) -> repeat(h,1,1) -> (h,w,d)  # 每行复制一份列 embedding
        # y_emb.unsqueeze(1): (h,1,d) -> repeat(1,w,1) -> (h,w,d)  # 每列复制一份行 embedding
        # cat 在最后维拼接 -> (h,w,2d)
        # permute(2,0,1) -> (2d,h,w)
        # unsqueeze(0) -> (1,2d,h,w)
        # repeat(B,1,1,1) -> (B,2d,h,w)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos  # (B,2*num_pos_feats,H,W)


def build_position_encoding(args):
    # N_steps：每个方向（x 或 y）的 pos feats 数
    # hidden_dim 通常是 transformer 的通道数（例如 256）
    # 因为最后会拼 y 和 x，所以 N_steps = hidden_dim/2
    N_steps = args.hidden_dim // 2

    # 支持两类 position embedding：sine / learned
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        # normalize=True：把坐标归一化到 [0,2*pi]，更通用
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        # 如果传入不支持的类型，直接报错
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding  # 返回一个 nn.Module，可直接对 NestedTensor 调用 forward
