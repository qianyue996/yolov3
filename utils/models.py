"""训练数据流中的核心数据模型及坐标转换工具。

数据流向（附形状说明）：
  原始图片 + 标注文本
       ↓
  RawTargets: (PIL.Image,  N×5 float32  像素坐标 x1y1x2y2class)
       ↓  yolo_collate_fn 中的 TransFormer
  TransformedBatch: (3,416,416) float32  归一化像素值
       + (N,5) float32  xyxy 归一化到 [0,1]（除以 416）
       ↓  YOLOLOSS.__call__ 内 xyxy2xywh
  FeatureTargets: (N,5) float32  cx,cy,w,h 归一化到当前特征图网格（stride=8/16/32）
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch


class RawTargets(NamedTuple):
    """Dataset.__getitem__ 返回的原始标注，单位：像素。"""
    boxes: torch.Tensor  # (N, 5)  [x1, y1, x2, y2, class_id]


class TransformedBatch(NamedTuple):
    """yolo_collate_fn 的输出，进入模型前的最终形态。"""
    images: torch.Tensor      # (B, 3, 416, 416)  ToTensor + Normalize
    targets: list[torch.Tensor]  # list of (Ni, 5)  xyxy ∈ [0,1]，class_id


def xyxy2xywh(
    targets: list[torch.Tensor], feat_h: int, feat_w: int
) -> list[torch.Tensor]:
    """将归一化 xyxy 格式的标注转换为 grid 坐标系下的 cx,cy,w,h 格式。

    targets:  每个元素 (Ni, 5)  xyxy ∈ [0,1]
    feat_h:   当前特征图高度（dim2 = 行数 = y 轴，如 13/26/52）
    feat_w:   当前特征图宽度（dim3 = 列数 = x 轴，如 13/26/52）
    返回:     每个元素 (Ni, 5)  cx,cy,w,h ∈ grid 单位
    """
    _targets = []
    for _target in targets:
        target = copy.deepcopy(_target)
        # 模型输出为 (B, nA, H, W, C)：dim2=H=行=y 轴，dim3=W=列=x 轴。
        # 因此 x 坐标沿 dim3（共 feat_w 个 cell），y 坐标沿 dim2（共 feat_h 个 cell）。
        target[:, [0, 2]] = target[:, [0, 2]] * feat_w
        target[:, [1, 3]] = target[:, [1, 3]] * feat_h
        x = ((target[:, 0] + target[:, 2]) / 2).unsqueeze(1)
        y = ((target[:, 1] + target[:, 3]) / 2).unsqueeze(1)
        w = (target[:, 2] - target[:, 0]).unsqueeze(1)
        h = (target[:, 3] - target[:, 1]).unsqueeze(1)
        c = target[:, 4].unsqueeze(1)
        target = torch.cat([x, y, w, h, c], dim=1)
        _targets.append(target)
    return _targets
