"""YOLOv3 公共解码逻辑。

将模型原始输出解码为 grid 单位的 (cx, cy, w, h) 坐标，
供 loss.py 和 postprocess.py 复用，消除重复代码。
"""

import torch


def decode_preds(
    pred: torch.Tensor,
    anchors: torch.Tensor,
    feat_h: int,
    feat_w: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """将模型原始输出解码为 grid 单位的 (cx, cy, w, h) 坐标。

    Args:
        pred: 模型原始输出 (B, N_anchors, H, W, 5+C)
        anchors: 当前层的 anchor 宽高，**grid 单位** (N_anchors, 2)
        feat_h: 当前特征图高度（dim2 = 行数 = y 轴，如 13/26/52）
        feat_w: 当前特征图宽度（dim3 = 列数 = x 轴，如 13/26/52）
        device: 计算设备

    Returns:
        (cx, cy, w, h, grid_x, grid_y)，均为 (B, N_anchors, H, W) 的 grid 单位张量
    """
    cx = pred.sigmoid()[..., 0] * 2 - 0.5
    cy = pred.sigmoid()[..., 1] * 2 - 0.5
    w = (pred[..., 2].sigmoid() * 2) ** 2
    h = (pred[..., 3].sigmoid() * 2) ** 2

    # 模型输出 (B, nA, H, W, C): dim2=H=行=y, dim3=W=列=x
    # grid_y 沿 dim0(=dim2=H 行) 递增, grid_x 沿 dim1(=dim3=W 列) 递增
    grid_y, grid_x = torch.meshgrid(
        torch.arange(feat_h, device=device),
        torch.arange(feat_w, device=device),
        indexing="ij",
    )

    anchor_w = anchors[:, 0].view(1, -1, 1, 1).expand_as(w)
    anchor_h = anchors[:, 1].view(1, -1, 1, 1).expand_as(h)

    cx = cx + grid_x
    cy = cy + grid_y
    w = w * anchor_w
    h = h * anchor_h

    # 边界 clamp：cx 沿 dim3=x(共 feat_w 格), cy 沿 dim2=y(共 feat_h 格)
    cx = cx.clamp(min=0, max=feat_w - 1)
    cy = cy.clamp(min=0, max=feat_h - 1)

    return cx, cy, w, h, grid_x, grid_y
