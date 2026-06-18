# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved  # 版权声明（原 DETR 系列代码风格）
"""
Backbone modules.  # 文件用途：定义 backbone（ResNet）和位置编码 joiner
"""

from collections import OrderedDict  # OrderedDict：有序字典（这里基本没用到，保留兼容老版本代码）
from typing import Dict, List  # 类型标注：Dict/List

import os  # 用于判断权重路径是否存在等
import torch  # PyTorch 主库
import torch.nn.functional as F  # 常用函数式接口（interpolate 等）
import torchvision  # torchvision.models 提供 ResNet
from torch import nn  # nn.Module 等
from torchvision.models._utils import IntermediateLayerGetter  # 从 backbone 中“抓取中间层输出”的工具

from util.misc import NestedTensor, is_main_process  # NestedTensor：tensor+mask；is_main_process：仅主进程打印
from .position_encoding import build_position_encoding  # 构建位置编码模块


def safe_torch_load(path, map_location="cpu"):
    """
    PyTorch 2.6+ 默认 weights_only=True，会导致旧 .tar / 含pickle结构的权重加载失败。
    这里强制 weights_only=False（若老版本torch不支持该参数则自动回退）。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # 新版 torch：显式允许加载完整对象
    except TypeError:
        return torch.load(path, map_location=map_location)  # 旧版 torch：没有 weights_only 参数就回退到默认


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):  # 防御：如果不是 dict（比如直接是 nn.Module），原样返回
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):  # 若不含 DDP 的 "module." 前缀，无需处理
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}  # 去掉最前面的 "module."（只替换一次）


def load_backbone_state_dict(backbone_path: str) -> Dict[str, torch.Tensor]:
    """
    兼容常见权重格式：
      1) 直接是 state_dict
      2) {'state_dict': ...}
      3) {'model': ...}
      4) {'net': ...}
      5) {'backbone': ...}
    """
    obj = safe_torch_load(backbone_path, map_location="cpu")  # 从磁盘读取 checkpoint 到 CPU（避免 GPU 占用）
    if isinstance(obj, dict):  # checkpoint 常见形式是 dict
        for key in ["state_dict", "model", "net", "backbone"]:  # 兼容不同 repo 的 key 命名
            if key in obj and isinstance(obj[key], dict):  # 找到真正的 state_dict 所在字段
                obj = obj[key]  # 取出内部 dict
                break  # 找到就停
    if not isinstance(obj, dict):  # 若仍不是 dict，说明格式不支持
        raise RuntimeError(f"[Backbone] Unsupported checkpoint format: {type(obj)} from {backbone_path}")  # 报错提示
    obj = _strip_module_prefix(obj)  # 处理 DDP 保存时可能出现的 module. 前缀
    return obj  # 返回标准 state_dict（key->tensor）


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    # 冻结 BN：不更新 running_mean/var，也不训练 weight/bias（通过 buffer 固定）
    # 好处：更稳定，尤其在小 batch 或检测任务里常用（DETR 标配）。
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    # 说明：rsqrt 前加 eps，防止数值问题（NaN）
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()  # 初始化父类
        self.register_buffer("weight", torch.ones(n))  # BN 的 gamma（固定为 1），作为 buffer（不训练）
        self.register_buffer("bias", torch.zeros(n))  # BN 的 beta（固定为 0）
        self.register_buffer("running_mean", torch.zeros(n))  # 固定的均值
        self.register_buffer("running_var", torch.ones(n))  # 固定的方差

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'  # 标准 BN 会带这个计数器
        if num_batches_tracked_key in state_dict:  # FrozenBN 不需要它
            del state_dict[num_batches_tracked_key]  # 删掉，避免 load_state_dict 时报 unexpected key

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,  # 继续交给父类正常加载 buffer
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)  # 变形以便广播到 N,C,H,W
        b = self.bias.reshape(1, -1, 1, 1)  # 同上
        rv = self.running_var.reshape(1, -1, 1, 1)  # 方差 reshape
        rm = self.running_mean.reshape(1, -1, 1, 1)  # 均值 reshape
        eps = 1e-5  # 数值稳定项
        scale = w * (rv + eps).rsqrt()  # scale = gamma / sqrt(var+eps)
        bias = b - rm * scale  # bias = beta - mean * scale
        return x * scale + bias  # BN 推理公式（固定参数，不更新统计量）


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()  # 初始化 nn.Module
        for name, parameter in backbone.named_parameters():  # 遍历 ResNet 所有参数
            # 如果不训练 backbone，或者不是 layer2/3/4（即只允许训练高层），就冻结参数
            if (not train_backbone) or (('layer2' not in name) and ('layer3' not in name) and ('layer4' not in name)):
                parameter.requires_grad_(False)  # 关闭梯度：训练时不会更新

        if return_interm_layers:  # 是否返回中间层特征（FPN 风格）
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}  # 多层输出
        else:
            return_layers = {"layer4": "0"}  # 只输出最高层（检测里常用，省内存）

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 创建“抓取中间层输出”的 wrapper
        self.num_channels = num_channels  # 记录 backbone 输出通道数（resnet50=2048 等）

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)  # 把图像 tensor 喂进 backbone，得到 dict：{layer_name: feature}
        out: Dict[str, NestedTensor] = {}  # 准备输出 dict（同样用 NestedTensor 包装 feature+mask）

        for name, x in xs.items():  # 遍历每个层输出
            m = tensor_list.mask  # 原始输入的 mask（padding 区域为 True/False 取决于你实现）
            assert m is not None  # 必须存在 mask
            # ✅ 用 nearest 更稳定（mask是离散量）
            mask = F.interpolate(m[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]
            # 解释：
            # 1) m[None] 增加 batch 维到 (1,B,H,W) 或 (1,H,W)（取决于 NestedTensor 实现）
            # 2) float() 是为了 interpolate
            # 3) nearest：mask 这种 0/1 离散值只能用最近邻，避免插值产生 0.3/0.7
            # 4) 转回 bool，并去掉前面加的维度 [0]

            out[name] = NestedTensor(x, mask)  # 用对齐后的 mask 包装每一层 feature

        return out  # 返回 dict：{ "0": NestedTensor(feature, mask), ... }


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""  # ResNet + FrozenBN 的 backbone 封装
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 backbone_path: str):

        # ✅ torchvision 0.13+ 推荐 weights=None 替代 pretrained=False（避免 deprecated warning）
        try:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],  # 是否在最后 stage 用 dilation 替换 stride
                weights=None,  # 不下载/不使用 torchvision 内置权重（你用本地权重加载）
                norm_layer=FrozenBatchNorm2d  # 用 FrozenBN 替换默认 BN
            )
        except TypeError:
            # 兼容旧 torchvision（没有 weights 参数）
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],  # 同上
                pretrained=False,  # 旧版接口：不加载预训练
                norm_layer=FrozenBatchNorm2d  # 同上
            )

        # ✅ 加载本地权重（兼容 tar/多key/module前缀）
        if backbone_path and os.path.isfile(backbone_path):  # 只有路径存在才加载
            sd = load_backbone_state_dict(backbone_path)  # 从文件解析出 state_dict
            missing, unexpected = backbone.load_state_dict(sd, strict=False)  # strict=False：允许 key 不完全匹配
            if is_main_process() and (len(missing) > 0 or len(unexpected) > 0):  # 仅主进程打印，避免多卡刷屏
                print(f"[Backbone] Loaded {backbone_path} with strict=False")  # 告知使用 strict=False
                if len(missing) > 0:
                    print(f"[Backbone] Missing keys (first 30): {missing[:30]}")  # 缺失参数（前 30 个）
                if len(unexpected) > 0:
                    print(f"[Backbone] Unexpected keys (first 30): {unexpected[:30]}")  # 多余参数（前 30 个）
        else:
            if is_main_process():
                print(f"[Backbone] WARNING: backbone_path not found: {backbone_path}, using random init.")
                # 路径不存在：Backbone 随机初始化（通常会影响效果）

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048  # 根据 resnet 深度确定输出通道数
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)  # 交给 BackboneBase 封装


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)  # 把两个模块按顺序放进 Sequential：self[0]=backbone, self[1]=pos

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # 先过 backbone，得到 dict：各层 NestedTensor
        out: List[NestedTensor] = []  # 按顺序收集 feature
        pos = []  # 按顺序收集 position embedding
        for name, x in xs.items():  # 遍历各层输出（dict 的遍历顺序取决于 IntermediateLayerGetter 的构造）
            out.append(x)  # 保存该层特征
            pos.append(self[1](x).to(x.tensors.dtype))  # 计算该层位置编码，并转成和 feature 相同 dtype（amp 下重要）
        return out, pos  # DETR 风格：返回 features 列表 + pos 列表


def build_backbone(args):
    position_embedding = build_position_encoding(args)  # 根据 args.position_embedding 构建 sine/learned 等位置编码
    train_backbone = args.lr_backbone > 0  # 是否训练 backbone：lr_backbone>0 才训练（否则全部冻结）
    return_interm_layers = False  # 这里固定为 False：只输出 layer4（最深层特征）
    backbone = Backbone(
        args.backbone,  # resnet50 / resnet101 ...
        train_backbone,  # 是否训练 backbone
        return_interm_layers,  # 是否输出中间层
        args.dilation,  # 是否 dilation
        args.backbone_path  # 本地权重路径
    )
    model = Joiner(backbone, position_embedding)  # 把 backbone 和 position embedding 组合成一个模块
    model.num_channels = backbone.num_channels  # 给 Joiner 挂一个 num_channels，供后续构建 transformer 使用
    return model  # 返回 Joiner(backbone+pos)
