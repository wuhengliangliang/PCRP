# models/sam3_wrapper.py
# -*- coding: utf-8 -*-  # UTF-8 编码，便于中文注释

from __future__ import annotations  # 允许类型标注引用尚未定义的类名（推迟解析）

import os          # 路径、文件判断
import inspect     # 读函数签名 inspect.signature / 判断函数类型
import importlib   # 动态 import 模块
import pkgutil     # 扫包内子模块（walk_packages）
from contextlib import nullcontext  # 无事可做的上下文（CPU 情况下替代 autocast）
from typing import Optional, Callable, Any, Dict, Tuple, Union, List  # 类型标注

import torch
import torch.nn.functional as F


# -------------------------
# PyTorch 2.6+ safe torch.load
# -------------------------
def safe_torch_load(path: str, map_location="cpu"):
    """
    PyTorch 2.6+ 默认 torch.load 会倾向于 weights_only=True（安全策略变化），
    这会导致“老格式 tar / 含 pickle 对象”加载失败。
    因此这里强制 weights_only=False；如果当前 torch 版本不支持该参数，就回退到老调用。

    ⚠️安全提醒：weights_only=False 允许反序列化任意对象，
    所以只在你信任 ckpt 来源时使用（官方/自己训练/可信来源）。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 老版本 torch 没有 weights_only 这个参数
        return torch.load(path, map_location=map_location)


def _rank0_print(msg: str):
    """
    只在 DDP 的 rank0 打印，避免多卡重复刷屏。
    若未初始化分布式，则正常 print。
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                print(msg, flush=True)
            return
    except Exception:
        pass
    print(msg, flush=True)


def _maybe_denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    尝试把 ImageNet normalize 过的图（均值方差归一）“还原”回 [0,1]。
    目的：SAM3 可能期望输入像素是 0~1，再做它自己的预处理。
    这里用一个启发式判断：如果 min/max 看起来不像 0~1，就按 ImageNet mean/std 反归一化。
    """
    # 如果不是 float 类型，强制转 float（便于 min/max 和后续运算）
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x = x.float()

    # 启发式判断：如果值域明显超出 [0,1]，说明可能做过 normalize
    if x.min() < -0.1 or x.max() > 1.3:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = x * std + mean  # 反归一化：x = x*std + mean

    # 最后 clamp 到 [0,1]，防止数值越界
    return x.clamp(0.0, 1.0)


def _cxcywh_to_xyxy_norm(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """
    将 DETR 常用的 (cx,cy,w,h) 归一化框 -> (x0,y0,x1,y1) 归一化框
    输入/输出都是 [0,1] 范围。

    boxes_cxcywh: (B,4) 其中每行是 cx,cy,w,h
    return:       (B,4) 其中每行是 x0,y0,x1,y1
    """
    cx, cy, w, h = boxes_cxcywh.unbind(dim=-1)  # 拆成四个 (B,)
    x0 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y0 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x1 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy + 0.5 * h).clamp(0.0, 1.0)
    return torch.stack([x0, y0, x1, y1], dim=-1)


def _resolve_sam3_root(sam3_pkg) -> Optional[str]:
    """
    给定 import sam3 得到的 package 对象，尝试推断“repo 根目录”：
    sam3_pkg.__file__ 通常是 .../sam3/sam3/__init__.py
    pkg_dir   = .../sam3/sam3
    repo_root = .../sam3
    """
    sam3_file = getattr(sam3_pkg, "__file__", None)
    if not sam3_file:
        return None
    pkg_dir = os.path.dirname(os.path.abspath(sam3_file))          # .../sam3/sam3
    repo_root = os.path.abspath(os.path.join(pkg_dir, os.pardir))  # .../sam3
    return repo_root


# -------------------------
# Robust builder discovery
# -------------------------
def _score_builder_name(fn_name: str) -> int:
    """
    给 builder 函数名打分：越像“构建 SAM3 image model”的函数，分越高。
    用于候选函数太多时挑最可能的那个。
    """
    n = fn_name.lower()
    s = 0
    if "build" in n:
        s += 5
    if "sam3" in n:
        s += 10
    if "image" in n:
        s += 6
    if "model" in n:
        s += 4
    # 明确命中优先名字，强力加分
    if fn_name in ("build_sam3_image_model", "build_sam3", "build_sam3_model"):
        s += 100
    return s


def _iter_candidate_modules(sam3_pkg) -> List[str]:
    """
    返回“可能包含 builder 的模块名列表”。
    先列一批常见模块名，再用 pkgutil.walk_packages 扫 sam3 包下子模块补充。
    """
    common = [
        "sam3.model_builder",
        "sam3.builder",
        "sam3.build",
        "sam3.build_sam3",
        "sam3.sam3_builder",
        "sam3.model.model_builder",
        "sam3.model.builder",
        "sam3.modeling.model_builder",
        "sam3.modeling.builder",
        "sam3.models.model_builder",
    ]

    mods: List[str] = []
    for m in common:
        mods.append(m)

    # 扫描 sam3 包下所有子模块，只把名字里含 builder/build/modeling 的加入候选
    try:
        for m in pkgutil.walk_packages(sam3_pkg.__path__, sam3_pkg.__name__ + "."):
            name = m.name
            low = name.lower()
            if any(k in low for k in ["builder", "build", "model_builder", "modeling"]):
                mods.append(name)
    except Exception:
        pass

    # 去重保序
    seen = set()
    out = []
    for m in mods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ⚠️ 下面这几行 import 与文件顶部重复（功能不受影响，只是重复了）
import importlib
import inspect
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec

def _load_module_from_path(mod_name: str, file_path: str):
    """
    从具体文件路径加载 python module（绕开 PYTHONPATH/import 的限制）。
    主要用于：你的 sam3 repo 结构比较特殊，正常 import 找不到 model_builder 时的兜底。
    """
    loader = SourceFileLoader(mod_name, file_path)  # 指定从 file_path 加载
    spec = spec_from_loader(mod_name, loader)       # 构造 module spec
    module = module_from_spec(spec)                 # 创建 module 对象
    loader.exec_module(module)                      # 执行 module 代码
    return module

def _import_model_builder_module(sam3_pkg):
    """
    优先尝试标准 import:
      - sam3.model_builder
      - sam3.sam3.model_builder
    如果都失败，再通过 repo_root/sam3/model_builder.py 的文件路径强制加载。
    """
    # 1) 正常 import
    for name in ["sam3.model_builder", "sam3.sam3.model_builder"]:
        try:
            return importlib.import_module(name)
        except Exception:
            pass

    # 2) 强制从文件加载
    sam3_root = _resolve_sam3_root(sam3_pkg)
    if sam3_root is None:
        raise RuntimeError("无法定位 sam3 repo_root，sam3_pkg.__file__ 为空？")

    mb_path = os.path.join(sam3_root, "sam3", "model_builder.py")  # /.../sam3/sam3/model_builder.py
    if not os.path.exists(mb_path):
        raise RuntimeError(f"找不到 model_builder.py: {mb_path}")

    return _load_module_from_path("_sam3_model_builder_from_path", mb_path)

def _pick_builder(sam3_pkg):
    """
    从 sam3 的 model_builder 模块里挑一个“builder 函数”。
    先按 preferred 名称列表找，找不到再扫描所有可能函数名，用启发式打分挑一个。
    """
    mb = _import_model_builder_module(sam3_pkg)

    # 常见命名优先级（先按这些名字找）
    preferred = [
        "build_sam3_image_model",
        "build_sam3_model",
        "build_sam3",
        "build_model",
        "model_builder",
        "build",
        "builder",
    ]
    for n in preferred:
        fn = getattr(mb, n, None)
        if callable(fn):
            # 打印出函数签名，帮助你确认 pick 的是不是你想要的 builder
            try:
                print(f"[SAM3] picked builder: {mb.__name__}.{n}{inspect.signature(fn)}")
            except Exception:
                print(f"[SAM3] picked builder: {mb.__name__}.{n}(signature unavailable)")
            return fn

    # 兜底：扫描所有“像 builder 的函数”
    cands = []
    for k, v in vars(mb).items():
        # 只挑“函数”，避免把类 __call__ 等当成 builder
        if not (inspect.isfunction(v) or inspect.isbuiltin(v)):
            continue
        kk = k.lower()
        # 名字里既要像 build/builder，又要含 sam
        if ("build" in kk or "builder" in kk) and ("sam" in kk):
            cands.append((k, v))

    if not cands:
        # 打印“可疑 callable”帮助定位
        maybe = [k for k, v in vars(mb).items() if callable(v) and any(s in k.lower() for s in ["build", "builder", "sam"])]
        raise RuntimeError(
            f"在 sam3 的 model_builder 中仍找不到 builder。\n"
            f"model_builder file: {getattr(mb, '__file__', 'N/A')}\n"
            f"callables matched: {maybe[:200]}"
        )

    # 多个候选就打分选最像的
    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "sam3" in n: s += 10
        if "image" in n: s += 6
        if "model" in n: s += 4
        if "build" in n: s += 2
        return s

    cands.sort(key=lambda x: score(x[0]), reverse=True)
    pick_name, pick_fn = cands[0]
    try:
        print(f"[SAM3] picked builder: {mb.__name__}.{pick_name}{inspect.signature(pick_fn)}")
    except Exception:
        print(f"[SAM3] picked builder: {mb.__name__}.{pick_name}(signature unavailable)")
    return pick_fn


def _call_builder(
    build_fn: Callable[..., Any],
    *,
    ckpt_path: str,
    device: torch.device,
    bpe_path: str,
    resolution: int,
) -> Tuple[Any, bool]:
    """
    调用 builder（兼容不同 sam3 版本的参数名差异）

    返回：(out, builder_loaded)
    - out：builder 的返回值（可能是 model、(model,xxx)、dict）
    - builder_loaded=True：说明 builder 函数签名里包含 ckpt 参数，
      我们传了 ckpt_path，它大概率内部已经 load 权重了；
    - False：说明 builder 只是构建网络结构，权重需要我们后面手动 load_state_dict。
    """
    sig = inspect.signature(build_fn)  # 读取函数签名（参数列表）
    params = sig.parameters            # OrderedDict(name->Parameter)

    # mapping：将“不同 sam3 builder 可能用的参数名”映射到我们的输入
    mapping: Dict[str, Any] = {
        # ckpt path aliases
        "checkpoint": ckpt_path,
        "ckpt_path": ckpt_path,
        "checkpoint_path": ckpt_path,
        "model_ckpt": ckpt_path,
        "ckpt": ckpt_path,
        "weights": ckpt_path,
        "weights_path": ckpt_path,
        "model_path": ckpt_path,

        # device
        "device": device,

        # bpe path aliases
        "bpe_path": bpe_path,
        "bpe": bpe_path,
        "tokenizer_path": bpe_path,
        "vocab_path": bpe_path,

        # resolution aliases
        "resolution": resolution,
        "image_size": resolution,
        "img_size": resolution,
        "input_size": resolution,
    }

    kwargs: Dict[str, Any] = {}  # 最终传给 builder 的 kwargs
    builder_loaded = False       # 是否认为 builder 自己 load 了 ckpt

    # 只填入 builder 函数签名里真正存在的参数名
    for k, v in mapping.items():
        if k in params and v is not None:
            kwargs[k] = v

    # 如果签名里含 ckpt 参数名，说明 builder 可能自己负责加载
    if any(k in params for k in ["checkpoint", "ckpt_path", "checkpoint_path", "model_ckpt", "ckpt", "weights", "weights_path", "model_path"]):
        builder_loaded = True

    # 1) 先按 kwargs 调用
    try:
        out = build_fn(**kwargs) if len(kwargs) > 0 else build_fn()
        return out, builder_loaded
    except TypeError:
        # 2) 若 TypeError（参数不匹配），多轮 fallback：逐步删掉 resolution / bpe 等参数再试

        fallback_kwargs_list = []

        # 去掉 resolution 相关参数
        kw1 = dict(kwargs)
        for k in ["resolution", "image_size", "img_size", "input_size"]:
            kw1.pop(k, None)
        fallback_kwargs_list.append(kw1)

        # 再去掉 bpe 相关参数
        kw2 = dict(kw1)
        for k in ["bpe_path", "bpe", "tokenizer_path", "vocab_path"]:
            kw2.pop(k, None)
        fallback_kwargs_list.append(kw2)

        # 只留 ckpt（如果 builder 支持）
        kw3 = {}
        for k in ["checkpoint", "ckpt_path", "checkpoint_path", "model_ckpt", "ckpt", "weights", "weights_path", "model_path"]:
            if k in params:
                kw3[k] = ckpt_path
                break
        fallback_kwargs_list.append(kw3)

        # 依次尝试 fallback 组合
        for kw in fallback_kwargs_list:
            try:
                out = build_fn(**kw) if len(kw) > 0 else build_fn()
                return out, builder_loaded
            except Exception:
                continue

        # 3) positional args 再兜底：尝试 build_fn(ckpt_path) 或 build_fn()
        for args in [(ckpt_path,), ()]:
            try:
                out = build_fn(*args)
                return out, builder_loaded
            except Exception:
                continue

        # 全部失败，抛出更详细的错误，提示你贴 builder 签名
        raise RuntimeError(
            f"SAM3 builder 调用失败：{build_fn.__module__}.{build_fn.__name__}{sig}\n"
            f"我们尝试过 kwargs={kwargs} 以及多轮 fallback 仍失败。\n"
            "请把你 sam3 中 builder 函数定义（函数签名）贴出来，我给你精确对齐。"
        )


def _unwrap_model(out: Any) -> Any:
    """
    builder 的返回值不一定直接是 model：
    - 可能是 (model, aux)
    - 可能是 dict{'model':...} / {'sam3':...} 等
    这里统一抽出模型对象返回。
    """
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    if isinstance(out, dict):
        for k in ["model", "sam3", "image_model", "net"]:
            if k in out:
                return out[k]
    return out


# out_size 参数的类型别名：
# - None：输出尺寸与输入 images 的原尺寸一致
# - int：输出方形尺寸 out_size x out_size
# - (h,w)：输出为指定 (h,w)
_OutSize = Optional[Union[int, Tuple[int, int]]]


class Sam3BoxMaskTeacher:
    """
    这是一个“Teacher wrapper”（像 teacher model 的调用器）：

    输入：
      - images: (B,3,H,W)
      - boxes_cxcywh_norm: (B,4) 归一化框 [0,1]
    输出：
      - teacher_mask_prob: (B,1,out_H,out_W) in [0,1]

    额外增强功能：
      - self.last_best_scores: 每张图选中 mask 的 score
      - self.last_valids     : score>=conf_th 才认为该样本 teacher 有效，否则 mask 置 0
      - prompt_coord: norm 或 pixel（控制把 prompt 坐标传给 sam3 的制式）
      - 缓存 text_out（forward_text）提高速度
    """

    def __init__(
        self,
        ckpt_path: str,               # sam3 权重路径
        device: torch.device,         # teacher 跑在哪个设备
        bpe_path: Optional[str] = None,  # tokenizer/bpe 文件路径（sam3 text encoder 用）
        resolution: int = 1024,          # sam3 输入分辨率（你的版本强制 16 倍数）
        confidence_threshold: float = 0.0,  # mask 置信度阈值：低于则 mask 全 0
        autocast_dtype: torch.dtype = torch.bfloat16,  # AMP dtype（bf16 常用于 A100/4090 等）
        prompt_coord: str = "norm",   # prompt 坐标制式："norm"(0~1) 或 "pixel"(0~res-1)
        cache_text_out: bool = True,  # 是否缓存 text_out
    ):
        self.device = device
        self.resolution = int(resolution)

        # 你的 sam3 版本要求输入分辨率是 16 的倍数（通常 stem/patch stride 对齐）
        if self.resolution % 16 != 0:
            raise ValueError(f"[SAM3 Teacher] resolution 必须是 16 的整数倍，当前={self.resolution}。建议直接用 1024。")

        self.confidence_threshold = float(confidence_threshold)
        self.autocast_dtype = autocast_dtype
        self.prompt_coord = str(prompt_coord)
        assert self.prompt_coord in ["norm", "pixel"], "prompt_coord must be 'norm' or 'pixel'"

        # 记录每次 __call__ 的 best score / valid
        self.last_best_scores: List[float] = []
        self.last_valids: List[bool] = []

        # 导入 sam3 包（要求你的 PYTHONPATH 能找到 sam3）
        try:
            import sam3 as sam3_pkg
        except Exception as e:
            raise RuntimeError(
                "找不到 sam3 包。请确保你的 PYTHONPATH 指向 sam3 的父目录。\n"
                "例如（按你当前工程）：\n"
                "  export PYTHONPATH=/data_3/pl/miccai/PathVG-main/sam3:$PYTHONPATH\n"
                "或者把 sam3 目录加到 train.sh 里。\n"
                f"原始错误：{repr(e)}"
            )

        # 自动推断 sam3 repo 根目录，用于拼 assets 路径
        sam3_root = _resolve_sam3_root(sam3_pkg)

        # 如果没显式传 bpe_path，则默认用 sam3/assets 下的 bpe 文件
        if bpe_path is None:
            if sam3_root is None:
                raise RuntimeError(
                    "无法自动定位 sam3/assets，请显式传 --sam3_bpe_path。\n"
                    "例如：--sam3_bpe_path /data_3/pl/miccai/PathVG-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                )
            bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        # 选 builder 函数
        build_fn = _pick_builder(sam3_pkg)

        # 调 builder：尽量把 ckpt/device/bpe/resolution 传进去（不同版本参数名不同）
        out, builder_loaded = _call_builder(
            build_fn,
            ckpt_path=ckpt_path,
            device=device,
            bpe_path=bpe_path,
            resolution=self.resolution,
        )

        # 把 builder 返回值统一拆出 model
        self.model = _unwrap_model(out)
        self.model.to(device)  # 移到 device
        self.model.eval()      # teacher 模式：推理（禁 dropout 等）

        # 如果 builder 没有加载 ckpt（builder_loaded=False），我们手动 load_state_dict
        if not builder_loaded:
            ckpt = safe_torch_load(ckpt_path, map_location="cpu")

            # 尝试兼容不同 ckpt 字段：model/state_dict/net/weights...
            state = None
            if isinstance(ckpt, dict):
                for k in ["model", "state_dict", "model_state_dict", "net", "weights"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]
                        break
            if state is None:
                state = ckpt  # 兜底：假设 ckpt 本身就是 state_dict

            # 去掉 DDP 的 module. 前缀
            if isinstance(state, dict) and any(kk.startswith("module.") for kk in state.keys()):
                state = {kk.replace("module.", "", 1): vv for kk, vv in state.items()}

            # strict=False：允许缺 key 或多 key（不同版本权重键名不完全一致）
            missing, unexpected = self.model.load_state_dict(state, strict=False)

            # 打印 key mismatch 信息，方便你排查版本差异
            if len(unexpected) > 0:
                _rank0_print(f"[SAM3 Teacher] Unexpected keys: {len(unexpected)} (first 20) {unexpected[:20]}")
            if len(missing) > 0:
                _rank0_print(f"[SAM3 Teacher] Missing keys: {len(missing)} (first 20) {missing[:20]}")

        # 导入 sam3 内部工具：FindStage + interpolate
        # - FindStage：构建 grounding 的输入结构（含 boxes/points 等 prompt）
        # - interpolate：sam3 自己封装的插值函数（可能有特殊兼容）
        from sam3.model.data_misc import FindStage, interpolate
        self._FindStage = FindStage
        self._interpolate = interpolate

        # ✅ 缓存 text_out（加速）
        # sam3 的 grounding 需要 text encoder 输出，但我们只用固定 prompt ["visual"]，
        # 所以可以一次算好重复用，减少每次 teacher 调用的开销。
        self._cached_text_out = None
        self._cache_text_out = bool(cache_text_out)

        if self._cache_text_out:
            # autocast 上下文：CUDA 才启用 AMP；CPU 用 nullcontext()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                # forward_text(["visual"])：得到文本侧特征（可能是 dict）
                self._cached_text_out = self.model.backbone.forward_text(["visual"], device=self.device)

            # 确保缓存不带梯度（虽然 inference_mode 已经保证，但更稳）
            if isinstance(self._cached_text_out, dict):
                for k, v in list(self._cached_text_out.items()):
                    if torch.is_tensor(v):
                        self._cached_text_out[k] = v.detach()

        _rank0_print(
            f"[SAM3 Teacher] prompt_coord={self.prompt_coord}  "
            f"conf_th={self.confidence_threshold}  cache_text_out={self._cache_text_out}"
        )

    @torch.inference_mode()
    def _preprocess(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        将输入图像预处理到 sam3 的输入规范：
        images_01: (B,3,H,W) in [0,1]
        -> resize 到 (B,3,res,res)
        -> normalize 到 [-1,1]（(x-0.5)/0.5）
        """
        x = F.interpolate(images_01, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    @staticmethod
    def _resolve_out_hw(images: torch.Tensor, out_size: _OutSize) -> Tuple[int, int]:
        """
        决定 teacher mask 输出尺寸 out_H,out_W：
        - out_size=None：保持与输入 images 的 H,W 一致
        - out_size=int：输出正方形 out_size x out_size
        - out_size=(h,w)：输出指定尺寸
        """
        _, _, H, W = images.shape
        if out_size is None:
            return int(H), int(W)
        if isinstance(out_size, int):
            return int(out_size), int(out_size)
        return int(out_size[0]), int(out_size[1])

    def _maybe_to_pixel(self, boxes_in: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 prompt_coord 决定 prompt 用“归一化坐标”还是“像素坐标”：
        - norm：保持 [0,1]（你算出来的就是归一化）
        - pixel：乘以 (res-1) 映射到 [0,res-1]
        """
        if self.prompt_coord == "norm":
            return boxes_in, points

        # pixel 坐标：乘以 res-1
        s = float(self.resolution - 1)
        boxes_in = (boxes_in * s).clamp(0.0, s)
        points = (points * s).clamp(0.0, s)
        return boxes_in, points

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor, boxes_cxcywh_norm: torch.Tensor, out_size: _OutSize = None) -> torch.Tensor:
        """
        Teacher 推理主入口：
        输入：
          images: (B,3,H,W) 可能已经是 normalize 后的，也可能是 NestedTensor.tensors
          boxes_cxcywh_norm: (B,4) in [0,1]
        输出：
          masks: (B,1,out_H,out_W) in [0,1]
        """
        # 如果上游传的是 NestedTensor，取出真实 tensor
        if hasattr(images, "tensors"):
            images = images.tensors

        # 基本形状检查
        assert images.dim() == 4 and images.size(1) == 3, f"images should be (B,3,H,W), got {tuple(images.shape)}"
        assert boxes_cxcywh_norm.dim() == 2 and boxes_cxcywh_norm.size(1) == 4, f"boxes should be (B,4), got {tuple(boxes_cxcywh_norm.shape)}"

        # 移到 device，box clamp 到 [0,1]
        images = images.to(self.device, non_blocking=True)
        boxes_cxcywh_norm = boxes_cxcywh_norm.to(self.device, non_blocking=True).clamp(0.0, 1.0).float()

        # 决定输出 mask 的尺寸（默认与输入图像一致）
        out_H, out_W = self._resolve_out_hw(images, out_size)

        # 把可能 ImageNet normalize 过的图，尽量还原到 [0,1]
        images_01 = _maybe_denormalize_imagenet(images)

        # resize 到 sam3 输入 res，并归一化到 [-1,1]
        sam_in_all = self._preprocess(images_01)  # (B,3,res,res)

        # autocast：CUDA 用 bf16/fp16，CPU 不用
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

        # text_out：如果缓存了就直接用；否则每次重新 forward_text
        if self._cached_text_out is not None:
            text_out = self._cached_text_out
        else:
            with autocast_ctx:
                text_out = self.model.backbone.forward_text(["visual"], device=self.device)

        # 收集每个样本的输出 mask、score、valid
        out_masks = []
        scores: List[float] = []
        valids: List[bool] = []

        B = sam_in_all.size(0)
        for i in range(B):
            # 逐张图片跑 sam3 grounding（你这里是 for 循环逐样本推理）
            sam_in = sam_in_all[i:i + 1]            # (1,3,res,res)
            box_cxcywh = boxes_cxcywh_norm[i:i + 1] # (1,4)

            # cxcywh -> xyxy（归一化）
            box_xyxy = _cxcywh_to_xyxy_norm(box_cxcywh)     # (1,4) in [0,1]

            # sam3 的 prompt 接口要求 boxes 的形状通常是 (B, N, 4)，这里 N=1
            boxes_in = box_xyxy.view(1, 1, 4).contiguous()  # (1,1,4)

            # points 取 box center：形状 (1,1,2)
            points = box_cxcywh[:, 0:2].view(1, 1, 2).contiguous()  # (1,1,2)

            # 根据 prompt_coord 转换坐标制式（norm / pixel）
            boxes_in, points = self._maybe_to_pixel(boxes_in, points)

            # prompt mask/label：告诉 sam3 哪些 box/point 是有效的
            boxes_mask  = torch.ones((1, 1), dtype=torch.bool, device=self.device)  # (1,1)
            boxes_label = torch.ones((1, 1), dtype=torch.long, device=self.device)  # (1,1) label=1（具体语义由 sam3 定义）
            points_mask = torch.ones((1, 1), dtype=torch.bool, device=self.device) # (1,1)

            with autocast_ctx:
                # 1) 前向图像 backbone
                backbone_out = self.model.backbone.forward_image(sam_in)

                # 2) 把 text_out 合并进去（sam3 的 forward_grounding 可能需要 backbone_out 同时含图像+文本特征）
                if isinstance(text_out, dict):
                    backbone_out.update(text_out)

                # 3) sam3 版本要求模型具备 _get_dummy_prompt（用于构建 geometric_prompt 容器）
                if not hasattr(self.model, "_get_dummy_prompt"):
                    raise RuntimeError("当前 sam3 模型对象没有 _get_dummy_prompt()，请对齐 sam3 版本。")

                geometric_prompt = self.model._get_dummy_prompt()

                # 将 box/point prompt 加入 prompt 容器（API 兼容不同版本：有 append_boxes/append_points 就调用）
                if hasattr(geometric_prompt, "append_boxes"):
                    geometric_prompt.append_boxes(boxes_in, boxes_mask)
                if hasattr(geometric_prompt, "append_points"):
                    geometric_prompt.append_points(points, points_mask)

                # 4) 构造 FindStage：grounding 输入结构（包含 boxes/points 等）
                find_stage = self._FindStage(
                    img_ids=torch.zeros((1,), device=self.device, dtype=torch.long),   # 这里固定为 0
                    text_ids=torch.zeros((1,), device=self.device, dtype=torch.long),  # 这里固定为 0（因为 text=["visual"]）
                    input_boxes=boxes_in,
                    input_boxes_mask=boxes_mask,
                    input_boxes_label=boxes_label,
                    input_points=points.squeeze(1),          # (1,2)（你的 sam3 FindStage 可能期望这样）
                    input_points_mask=points_mask.squeeze(1),
                )

                # 5) 调用 grounding：得到多个候选 mask + 对应 logits/score
                outputs = self.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,  # 推理时没有 target
                )

                # pred_masks: (1,N,h,w) logits（未 sigmoid）
                pred_masks = outputs["pred_masks"]

                # pred_logits: (1,N,1) 或 (1,N,C)（你这里只关心“存在/匹配得分”）
                pred_logits = outputs["pred_logits"]

                # presence_logit_dec：可选的“presence”置信度（某些版本有）
                presence = outputs.get("presence_logit_dec", None)

                # 先对 pred_logits 做 sigmoid 得到概率
                probs = pred_logits.sigmoid()

                # 若最后一维是 1，则 squeeze -> (1,N)
                if probs.dim() == 3 and probs.size(-1) == 1:
                    probs = probs.squeeze(-1)  # (1,N)

                # 若存在 presence 分支，则把 probs 再乘上 presence 的 sigmoid（进一步校准）
                if presence is not None:
                    if presence.dim() == 1:
                        probs = probs * presence.sigmoid().unsqueeze(1)
                    elif presence.dim() == 2:
                        probs = probs * presence.sigmoid()

                # 选择 score 最大的候选 mask
                best_idx = int(probs[0].argmax().item())
                best_score = float(probs[0, best_idx].item())

                # 是否有效：score >= 阈值
                is_valid = (best_score >= self.confidence_threshold)

                # 取出最佳 mask 的 logits: (h,w)
                best_mask_logits = pred_masks[0, best_idx]

                # 插值到 out_H,out_W，再 sigmoid 转成概率 mask
                best_mask_prob = self._interpolate(
                    best_mask_logits.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                    (out_H, out_W),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()  # (1,1,out_H,out_W) in [0,1]

                # 若无效：直接返回全 0 mask（相当于“teacher 不提供监督”）
                if not is_valid:
                    best_mask_prob = torch.zeros_like(best_mask_prob)

            # 保存本样本结果
            out_masks.append(best_mask_prob)
            scores.append(best_score)
            valids.append(bool(is_valid))

        # 保存最近一次调用的 score/valid，便于外部做日志统计或可靠性建模
        self.last_best_scores = scores
        self.last_valids = valids

        # 拼成 batch 输出：(B,1,out_H,out_W)
        return torch.cat(out_masks, dim=0)
