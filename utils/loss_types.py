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
    """每层 loss 计算后产出的 7 个归一化指标与诊断信息。"""

    loss_loc: float  # 定位 GIoU 损失（理想值 0.0，初期 ~2.5，收敛 < 0.4）
    loss_conf: float  # 置信度 Focal 损失（理想值 0.0，初期 ~3.5，收敛 < 0.08）
    loss_cls: float  # 分类 BCE 损失（理想值 0.0，初期 ~0.8，收敛 < 0.05）
    center_diff: float  # 中心点误差（grid 单位，理想 0.0，初期 20~30，收敛 < 0.8）
    wh_diff: float  # 宽高误差（grid 单位，理想 0.0，初期 15~25，收敛 < 1.2）
    conf_diff: (
        float  # |sigmoid(conf) - target| 绝对误差（理想 0.0，初期 ~0.5，收敛 < 0.05）
    )
    n_pos: int  # 本层分配的正样本数量
