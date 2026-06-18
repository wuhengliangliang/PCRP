# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np

import torchvision


def _parse_tv_version(v: str):
    """
    Parse torchvision version string robustly.

    Examples:
      - '0.15.2' -> (0, 15, 2)
      - '0.14.1+cu117' -> (0, 14, 1)
      - '0.8.2' -> (0, 8, 2)

    NOTE:
      DO NOT use torchvision.__version__[:3] (e.g. '0.15.2'[:3] == '0.1')
      which breaks version comparisons.
    """
    v = v.split("+")[0]
    parts = v.split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except Exception:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


_TV_VERSION = _parse_tv_version(torchvision.__version__)

# needed due to empty tensor bug in pytorch and old torchvision
if _TV_VERSION < (0, 7, 0):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def _dist_reduce_device() -> torch.device:
    """
    ✅ 关键：决定分布式通信时张量放在哪个 device 上
    - NCCL => CUDA (当前 rank 的当前卡)
    - GLOO/MPI => CPU
    - 未初始化 => CPU
    """
    if not (dist.is_available() and dist.is_initialized()):
        return torch.device("cpu")
    try:
        backend = dist.get_backend()
    except Exception:
        backend = None

    if backend == "nccl" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        dev = _dist_reduce_device()
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=dev)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    dev = _dist_reduce_device()

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(dev)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=dev, dtype=torch.long)
    size_list = [torch.tensor([0], device=dev, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list_int = [int(size.item()) for size in size_list]
    max_size = max(size_list_int)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=dev) for _ in range(world_size)]

    local_size_int = int(local_size.item())
    if local_size_int != max_size:
        padding = torch.empty((max_size - local_size_int,), dtype=torch.uint8, device=dev)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, t in zip(size_list_int, tensor_list):
        buf = t[:size].cpu().numpy().tobytes()
        data_list.append(pickle.loads(buf))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum

    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    dev = _dist_reduce_device()

    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            v = input_dict[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=dev)
            else:
                if v.device != dev:
                    v = v.to(dev)
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_v2(self, key, value, num):
        self.meters[key].update(value, num)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn_with_mask(batch):
    batch = list(zip(*batch))
    img = torch.stack(batch[0])
    mask = torch.stack(batch[1])
    samples = NestedTensor(img, mask)
    targets = tensor_dict_from_dict_list(batch[2])
    return tuple([samples, targets])


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def pin_memory(self):
        self.tensors = self.tensors.pin_memory()
        if self.mask is not None:
            self.mask = self.mask.pin_memory()
        return self

    def __repr__(self):
        return str(self.tensors)


class ListTensor(object):
    def __init__(self, tensors):
        self.tensors = tensors

    def pin_memory(self):
        self.tensors = [t.pin_memory() for t in self.tensors]
        return self


class TensorDict(object):
    def __init__(self, tensor_dict):
        self.tensor_dict = tensor_dict

    def pin_memory(self):
        for k in self.tensor_dict:
            self.tensor_dict[k] = self.tensor_dict[k].pin_memory()
        return self


def tensor_dict_from_dict_list(tensor_list: List[dict]):
    keys = [k for k in tensor_list[0].keys() if k != 'phrase']
    tensor_dict = dict()
    for k in keys:
        tensor_dict[k] = torch.stack([t[k] for t in tensor_list], dim=0)
    return TensorDict(tensor_dict)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Robust distributed init:
    - bind rank->GPU BEFORE init_process_group
    - support torchrun env + slurm env
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        ngpu = torch.cuda.device_count()
        args.gpu = args.rank % max(1, ngpu)
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return

    args.distributed = True

    if torch.cuda.is_available():
        args.dist_backend = 'nccl'
        torch.cuda.set_device(args.gpu)
    else:
        args.dist_backend = 'gloo'

    if not hasattr(args, "dist_url") or args.dist_url is None:
        args.dist_url = "env://"

    print('| distributed init (rank {}): {}, gpu={}'.format(
        args.rank, args.dist_url, args.gpu
    ), flush=True)

    timeout = datetime.timedelta(minutes=30)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timeout,
    )

    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    """
    if _TV_VERSION < (0, 7, 0):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
