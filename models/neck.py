import torch  # PyTorch 张量库
import torch.nn.functional as F  # 常用函数：softmax 等
from torch import nn  # nn.Module / Conv2d / Linear 等
from models.vis_dict import VisualDict  # 你的“视觉字典/量化器”（类似 VQ/VQ-VAE 的 codebook）


class PrototypeLearner(nn.Module):
    def __init__(self, num_tokens=2048, token_dim=768, hidden_dim=256, decay=0.1, max_decay=0.99):
        super().__init__()  # 初始化 Module 基类

        # =========================
        # 维度记录（用于 reshape/conv）
        # =========================
        self.hidden_dim = hidden_dim  # 输入特征通道（通常是 transformer hidden_dim，比如 256）
        self.token_dim = token_dim    # 原型/字典 token 维度（比如 768）

        # =========================
        # 视觉字典（原型池 / codebook）
        # num_tokens：字典大小（码本大小）
        # token_dim：每个原型的维度
        # decay：EMA 更新的衰减系数（具体取决于 VisualDict 实现）
        # =========================
        self.prototype = VisualDict(num_tokens=num_tokens, token_dim=token_dim, decay=decay)

        # =========================
        # 1x1 conv：通道映射
        # conv_head: hidden_dim -> token_dim，把输入特征投影到字典/原型空间
        # conv_tail: token_dim -> hidden_dim，把融合后的原型特征再投回模型 hidden_dim
        # =========================
        self.conv_head = nn.Conv2d(hidden_dim, token_dim, kernel_size=1, bias=False)
        self.conv_tail = nn.Conv2d(token_dim, hidden_dim, kernel_size=1, bias=False)

        # =========================
        # 是否做“位置对齐/线性对齐”
        # 这里的 pos_align 实际上不是“加位置编码”，而是对量化出来的 prototype 再做一次线性映射
        # 用于让量化输出的分布更适配当前网络的 token space
        # =========================
        self.pos_align = True
        if self.pos_align:
            self.pos_line = nn.Linear(token_dim, token_dim)  # 对 quantized_pt 做线性变换

        # =========================
        # 门控融合（gating）
        # 输入通道 token_dim*2（原型特征 + 原始图像特征拼接）
        # 输出通道 2：对应两个分支的权重（prototype 分支 / image 分支）
        # 用 softmax 归一化后做加权融合
        # =========================
        self.gate_fc = nn.Conv2d(token_dim * 2, 2, kernel_size=1, stride=1, bias=False)

    def forward(self, src: torch.tensor, h: int, w: int):
        """
        src: (HW, B, hidden_dim)   # 一般来自 transformer encoder 的 token 序列
        h,w: token 网格尺寸，满足 HW = h*w

        返回:
          embedded_pt: (HW, B, hidden_dim)  # 融合后的特征序列
          indices: (B*HW,) 或类似形状        # 每个 token 对应的字典索引（依 VisualDict 实现）
        """
        # ⚠️ 这里类型标注最好写 torch.Tensor，而不是 torch.tensor（后者是函数/构造器名）
        # 不影响运行，只是类型提示不严谨

        # 取出序列长度、batch、通道
        num_visu_token, bs, hidden_dim = src.size()  # num_visu_token 其实就是 HW

        # =========================
        # 1) 序列 -> feature map
        # src: (HW,B,C) -> (B,C,HW) -> (B,C,h,w)
        # =========================
        src = src.permute(1, 2, 0).contiguous().view(bs, -1, h, w)
        # 现在 src: (B, hidden_dim, h, w)

        # =========================
        # 2) 投影到 token_dim 空间（准备送入字典/原型量化）
        # =========================
        xq = self.conv_head(src)  # (B, token_dim, h, w)

        # =========================
        # 3) 拉平成 token 列表，喂给 VisualDict 做量化/查表
        # inputs_flatten: (B*h*w, token_dim)
        # =========================
        inputs_flatten = xq.view(-1, self.token_dim)

        # 保存一份“原始图像特征分支”（仍在 token_dim 空间）
        xq_img = xq  # (B, token_dim, h, w)

        # =========================
        # 4) 量化 / 原型检索
        # quantized_pt: (B*HW, token_dim)  # 每个 token 被替换为字典中最近的 prototype 向量（或 EMA 更新的向量）
        # indices:      (B*HW,)            # 每个 token 的字典索引
        # =========================
        quantized_pt, indices = self.prototype(inputs_flatten)

        # 可选：对 prototype 输出再做一层线性对齐（让 prototype 更适配当前 token 表征空间）
        if self.pos_align:
            quantized_pt = self.pos_line(quantized_pt)  # (B*HW, token_dim)

        # =========================
        # 5) prototype token 列表 -> prototype feature map
        # 目标：变回 (B, token_dim, h, w)，与 xq_img 对齐
        # =========================
        embedded_pt = quantized_pt.view(bs, num_visu_token, quantized_pt.size(-1))
        # embedded_pt: (B, HW, token_dim)

        embedded_pt = embedded_pt.permute(0, 2, 1).contiguous().view(bs, -1, h, w)
        # embedded_pt: (B, token_dim, h, w)

        # =========================
        # 6) 门控融合：prototype 分支 + image 分支
        # 拼接后过 gate_fc，输出两个通道，再 softmax 得到每个位置的两分支权重
        # =========================
        tmp_feat = torch.cat([embedded_pt, xq_img], dim=1)
        # tmp_feat: (B, 2*token_dim, h, w)

        tmp_score = F.softmax(self.gate_fc(tmp_feat), dim=1)
        # gate_fc 输出: (B,2,h,w)
        # softmax(dim=1) 后：
        # tmp_score[:,0,:,:] + tmp_score[:,1,:,:] = 1
        # 表示每个空间位置“prototype vs image”两分支的权重分配

        # prototype 分支权重（扩成 (B,1,h,w) 便于广播乘）
        emb_score = tmp_score[:, 0, :, :].unsqueeze(dim=1)  # (B,1,h,w)

        # image 分支权重
        img_score = tmp_score[:, 1, :, :].unsqueeze(dim=1)  # (B,1,h,w)

        # 按位置加权融合（每个 token 位置都有自己的融合比例）
        embedded_pt = embedded_pt * emb_score + xq_img * img_score
        # embedded_pt: (B, token_dim, h, w)

        # =========================
        # 7) 融合后的 token_dim 特征映射 -> hidden_dim
        # 返回给主干网络（通常 transformer 后续层）使用
        # =========================
        embedded_pt = self.conv_tail(embedded_pt)  # (B, hidden_dim, h, w)

        # =========================
        # 8) feature map -> 序列（回到 transformer 习惯的 (HW,B,C)）
        # flatten(2): (B, C, HW)
        # permute(2,0,1): (HW, B, C)
        # =========================
        return {
            "embedded_pt": embedded_pt.flatten(2).permute(2, 0, 1).contiguous(),  # (HW,B,hidden_dim)
            "indices": indices  # 字典索引（用于统计/可视化/损失等）
        }
