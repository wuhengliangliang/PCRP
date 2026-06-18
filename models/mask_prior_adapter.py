# models/mask_prior_adapter.py
# -*- coding: utf-8 -*-  # 文件编码声明（对中文注释安全）

from __future__ import annotations  # 允许在类型标注中写“尚未定义的类名”（推迟解析），避免循环引用问题

from dataclasses import dataclass  # dataclass：轻量数据结构，自动生成 __init__/__repr__ 等
from typing import Tuple, Optional  # 类型标注：Tuple 和 Optional

import torch  # PyTorch 张量
import torch.nn as nn  # nn.Module 等
import torch.nn.functional as F  # 函数式接口（interpolate）


@dataclass
class MaskPriorOut:
    # 输出结构：把两类结果打包返回，便于上层代码使用更清晰
    mask_bias: torch.Tensor    # (HW, B, 1)  供 transformer token 使用的 bias（按 token 展开）
    reliability: torch.Tensor  # (B, 1) in [0,1]  对应每个样本（B）的可靠性系数


class MaskPriorAdapter(nn.Module):
    """
    将 SAM3 mask_prob (B,1,H,W) 映射到 token 网格 bias (HW,B,1)，并用可靠性自适应缩放。

    背后意图：
    - SAM3 输出的是“像素/网格”级的 mask 概率 (H,W)
    - 你的视觉编码里通常有 token 网格 (h,w)（例如 20x20）
    - 所以要把 mask 概率 resize 到 token 网格，然后展平为 HW 个 token 的 bias
    - reliability 用来控制“这个 mask 先验你信不信”：越不可靠越缩小它的影响
    """

    def __init__(self, gamma_init: float = 1.0, learnable_gamma: bool = True):
        super().__init__()  # 初始化 nn.Module 基类

        # gamma：整体缩放因子，用来控制 mask prior 注入的强度
        # - learnable_gamma=True：训练时可学习（模型自己决定要用多强）
        # - learnable_gamma=False：固定常数（不训练）
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))  # 可学习参数，会出现在 model.parameters()
        else:
            # register_buffer：作为状态保存到模型里，但不参与梯度训练
            # persistent=False：不写入 state_dict（可选策略：避免 ckpt 里多一个无关量）
            self.register_buffer("gamma", torch.tensor(float(gamma_init)), persistent=False)

    def forward(
        self,
        mask_prob: torch.Tensor,             # (B,1,H,W) in [0,1]  来自 SAM3 的概率 mask（不是 logit）
        hw: Tuple[int, int],                 # (h,w) e.g. (20,20)  token 网格尺寸（目标分辨率）
        reliability: Optional[torch.Tensor] = None,  # (B,1) in [0,1] 每个样本的可靠性（可选）
    ) -> MaskPriorOut:
        # B：batch size
        B = mask_prob.size(0)

        # 目标 token 网格高宽，确保是 int
        h, w = int(hw[0]), int(hw[1])

        # 1) 把 SAM3 的 mask 从 (H,W) resize 到 token 网格 (h,w)
        # - bilinear：适用于概率图（连续值），不会像 nearest 那样块状
        # - align_corners=False：插值更稳定、也是很多视觉任务的常用设置
        # 输出 ms: (B,1,h,w)
        ms = F.interpolate(mask_prob, size=(h, w), mode="bilinear", align_corners=False)

        # 2) 将 (B,1,h,w) 变成 token 形式 (HW,B,1)，以便与 transformer 的 token 序列维度对齐
        # ms.flatten(2): (B,1,HW)  # 从第2维开始展平，把 (h,w) 合并成 HW
        # .permute(2,0,1): (HW,B,1) # 把 HW 放到最前面（对应 token 序列长度）
        # .contiguous(): 保证内存连续（有些后续 op/视图操作更安全/更快）
        bias = ms.flatten(2).permute(2, 0, 1).contiguous()

        # 3) 如果外部没有提供 reliability，则默认全信任（=1）
        # reliability shape 约定为 (B,1)，表示每个样本一个系数
        if reliability is None:
            reliability = torch.ones((B, 1), device=mask_prob.device, dtype=mask_prob.dtype)

        # 4) 将 bias 乘上 (reliability * gamma)
        # reliability.view(1,B,1)：把 (B,1) reshape 成可广播到 (HW,B,1) 的形状
        # self.gamma：全局缩放（学习/固定皆可）
        # 效果：每个样本的所有 token bias 都按该样本的可靠性缩放
        bias = bias * (reliability.view(1, B, 1) * self.gamma)

        # 5) 打包输出：mask_bias 给上游模块使用；reliability 也返回（便于日志/后续门控）
        return MaskPriorOut(mask_bias=bias, reliability=reliability)
