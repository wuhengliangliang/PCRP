import torch
from torchvision.ops.boxes import box_area
import math

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy2xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    """
    # 检查框是否包含 NaN 值
    if torch.isnan(boxes1).any() or torch.isnan(boxes2).any():
        print(f"存在 NaN 值：boxes1={boxes1}, boxes2={boxes2}")
        return torch.zeros(boxes1.size(0), boxes2.size(0), device=boxes1.device)  # 返回一个零矩阵，避免后续计算错误

    # 确保框的有效性：框的 x2 > x1 且 y2 > y1
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        print(f"无效的框，源框数据：{boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        print(f"无效的框，目标框数据：{boxes2}")
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


##############################################################
"""
    written by Wayne Tomas
    CIoU 实现（top）
"""
##############################################################

def ciou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    r"""Computes the Complete IoU loss as described in `"Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression" <https://arxiv.org/pdf/1911.08287.pdf>`_.
    """
    iou, union = box_iou(boxes1, boxes2)
    v = aspect_ratio_consistency(boxes1, boxes2)

    ciou_loss = 1 - iou + iou_penalty(boxes1, boxes2)

    # Check
    _filter = (v != 0) & (iou != 0)
    ciou_loss[_filter].addcdiv_(v[_filter], 1 - iou[_filter] + v[_filter])

    return ciou_loss


def aspect_ratio_consistency(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes the aspect ratio consistency from the complete IoU loss
    """
    v = aspect_ratio(boxes1).unsqueeze(-1) - aspect_ratio(boxes2).unsqueeze(-2)
    v.pow_(2)
    v.mul_(4 / math.pi**2)

    return v


def aspect_ratio(boxes: torch.Tensor) -> torch.Tensor:
    """Computes the aspect ratio of boxes
    """
    return torch.atan((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))


def iou_penalty(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Implements the penalty term for the Distance-IoU loss
    """
    # Diagonal length of the smallest enclosing box
    c2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2), device=boxes1.device)
    c2[..., 0] = torch.max(boxes1[:, 2].unsqueeze(-1), boxes2[:, 2].unsqueeze(-2))
    c2[..., 1] = torch.max(boxes1[:, 3].unsqueeze(-1), boxes2[:, 3].unsqueeze(-2))
    c2[..., 0].sub_(torch.min(boxes1[:, 0].unsqueeze(-1), boxes2[:, 0].unsqueeze(-2)))
    c2[..., 1].sub_(torch.min(boxes1[:, 1].unsqueeze(-1), boxes2[:, 1].unsqueeze(-2)))

    c2.pow_(2)
    c2 = c2.sum(dim=-1)

    # L2 - distance between box centers
    center_dist2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2), device=boxes1.device)
    center_dist2[..., 0] = boxes1[:, [0, 2]].sum(dim=1).unsqueeze(1)
    center_dist2[..., 1] = boxes1[:, [1, 3]].sum(dim=1).unsqueeze(1)
    center_dist2[..., 0].sub_(boxes2[:, [0, 2]].sum(dim=1).unsqueeze(0))
    center_dist2[..., 1].sub_(boxes2[:, [1, 3]].sum(dim=1).unsqueeze(0))

    center_dist2.pow_(2)
    center_dist2 = center_dist2.sum(dim=-1) / 4

    return center_dist2 / c2


##############################################################
"""
    written by Wayne Tomas
    CIoU 实现（tail）
"""
##############################################################
