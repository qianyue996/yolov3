"""训练数据流中的核心数据模型。"""

from __future__ import annotations

from typing import NamedTuple

import torch


# ── Loss 构建目标（build_targets 输出） ─────────────────────────────────────
class TargetBuild(NamedTuple):
    """build_targets 的返回值，与模型输出形状一一对应。"""

    y_true: torch.Tensor  # (B, 3, H, W, 5+C)
    noobj_mask: torch.Tensor  # (B, 3, H, W)  1=忽略的背景
    box_loss_scale: torch.Tensor  # (B, 3, H, W)  小物体放大系数


# ── 预测框解码（get_ignore 输出） ──────────────────────────────────────────
class PredDecode(NamedTuple):
    """get_ignore 的返回值，预测框在 grid 坐标系下。"""

    noobj_mask: torch.Tensor  # (B, 3, H, W)  更新后
    pred_boxes: torch.Tensor  # (B, 3, H, W, 4)  [cx, cy, w, h]，单位 grid cell


# ── Loss 标量指标 ───────────────────────────────────────────────────────────
class LayerMetrics(NamedTuple):
    """每层 loss 计算后产出的 6 个归一化指标。"""

    loss_loc: float  # GIoU loss，仅正样本，范围 ≥ 0
    loss_conf: float  # BCE(conf, obj_mask)，有效 cell，范围 ≥ 0
    loss_cls: float  # BCE(cls, targ_cls)，仅正样本，范围 ≥ 0
    center_diff: float  # 预测 vs GT 中心点误差（grid 单位），理想 0
    wh_diff: float  # 预测 vs GT 宽高误差（anchor 单位），理想 0
    conf_diff: float  # sigmoid(conf) vs obj_mask 的绝对误差，范围 [0,1]
    n_pos: int  # 正样本数量
