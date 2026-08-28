import torch
import torch.nn as nn

from .config import IMG_W
from .decode import decode_preds
from .models import xyxy2xywh

# 模型原始输出：每个 cell 3 个 anchor，每 anchor 输出 (5+C) 个 logit
# [ (B,3,H0,W0,5+C), (B,3,H1,W1,5+C), (B,3,H2,W2,5+C) ]
RawPredicts = list[torch.Tensor]


class YOLOLOSS:
    def __init__(self, model: nn.Module) -> None:
        self.device = next(model.parameters()).device
        self.stride = [8, 16, 32]
        self.anchors = torch.tensor(
            model.anchors, device=self.device, dtype=torch.float32
        )
        self.anchors_mask = model.anchors_mask
        self.class_name = model.class_names

        self.balance = [4, 1.0, 0.4]
        self.box_ratio = 0.05
        self.obj_ratio = 5
        self.cls_ratio = 1

    def __call__(
        self, predicts: RawPredicts, all_targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        """计算总 loss 并返回每层的详细指标字典，返回 (总loss, {layer0: {...}, layer1: {...}, layer2: {...}})。"""
        total_loss = torch.zeros((), device=self.device)
        detail = {}
        for layer_idx, pred in enumerate(predicts):
            bs = pred.shape[0]
            feat_h = pred.shape[2]
            feat_w = pred.shape[3]
            targets = xyxy2xywh(all_targets, feat_h, feat_w)
            anchors_mask = self.anchors_mask[layer_idx]

            y_true, noobj_mask, box_loss_scale = self.build_targets(
                bs, feat_h, feat_w, anchors_mask, pred, targets
            )
            noobj_mask, pred_boxes, grid_x, grid_y = self.get_ignore(
                bs, feat_h, feat_w, anchors_mask, pred, targets, noobj_mask
            )
            box_loss_scale = 2 - box_loss_scale

            obj_mask = y_true[..., 4] == 1
            n = torch.sum(obj_mask)

            layer_detail: dict = {}
            if n != 0:
                giou = self.box_giou(pred_boxes, y_true[..., :4], grid_x, grid_y)
                loss_loc = ((1 - giou) * box_loss_scale)[obj_mask].mean()

                center_diff = (
                    (pred_boxes[..., 0] - y_true[..., 0]).abs()
                    + (pred_boxes[..., 1] - y_true[..., 1]).abs()
                )[obj_mask].mean()
                wh_diff = (
                    (pred_boxes[..., 2] - y_true[..., 2]).abs()
                    + (pred_boxes[..., 3] - y_true[..., 3]).abs()
                )[obj_mask].mean()

                pred_cls = pred[..., 5:][obj_mask]
                targ_cls = y_true[..., 5:][obj_mask]
                loss_cls = nn.BCEWithLogitsLoss(reduction="mean")(pred_cls, targ_cls)
            else:
                loss_loc = torch.tensor(0.0, device=self.device)
                center_diff = torch.tensor(0.0, device=self.device)
                wh_diff = torch.tensor(0.0, device=self.device)
                loss_cls = torch.tensor(0.0, device=self.device)

            valid_mask = noobj_mask.bool() | obj_mask
            pred_conf_flat = pred[..., 4][valid_mask]
            conf_target = obj_mask.type_as(pred_conf_flat)[valid_mask]
            loss_conf = (
                focal_loss(
                    pred_conf_flat, conf_target, alpha=0.25, gamma=1.5, reduction="sum"
                )
                / max(1, n)
                / max(1, bs)
            )
            conf_diff = (pred_conf_flat.sigmoid() - conf_target).abs().mean()

            total_loss = total_loss + loss_loc * self.box_ratio
            total_loss = total_loss + loss_cls * self.cls_ratio
            total_loss = (
                total_loss + loss_conf * self.balance[layer_idx] * self.obj_ratio
            )

            layer_detail = {
                "loss_loc": loss_loc.item(),
                "loss_conf": loss_conf.item(),
                "loss_cls": loss_cls.item(),
                "center_diff": center_diff.item(),
                "wh_diff": wh_diff.item(),
                "conf_diff": conf_diff.item(),
                "n_pos": n.item(),
            }
            detail[f"layer{layer_idx}"] = layer_detail

        return total_loss, detail

    def build_targets(
        self,
        bs: int,
        feat_h: int,
        feat_w: int,
        anchors_mask: list[int],
        predict: torch.Tensor,
        targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为每个检测层构建 ground truth 张量 y_true，返回 (y_true, noobj_mask, box_loss_scale)。

        在全部 9 个 anchor 中选出全局最优 anchor，仅当最优 anchor 属于本层时才分配；
        歧义负样本由 get_ignore 处理。
        """
        y_true = torch.zeros_like(predict)
        noobj_mask = torch.ones(
            bs, len(anchors_mask), feat_h, feat_w, device=self.device
        )
        box_loss_scale = torch.zeros(
            bs, len(anchors_mask), feat_h, feat_w, device=self.device
        )
        stride = IMG_W / feat_h

        for b, target in enumerate(targets):
            if len(target) == 0:
                continue
            # target[:, 2:4] 在 grid 单位，乘 stride 转换到 416 像素尺度与全部 9 个 anchor 匹配
            target_wh_pixels = target[:, 2:4] * stride
            iou_all = compute_iou_with_anchors(target_wh_pixels, self.anchors)
            best_anchor_all = torch.argmax(iou_all, dim=-1)

            for t, best_a in enumerate(best_anchor_all):
                best_a_idx = int(best_a.item())
                # 只有当该目标的全局最佳 anchor 属于当前检测层时，才分配给本层
                if best_a_idx not in anchors_mask:
                    continue
                k = anchors_mask.index(best_a_idx)

                # target[t,0]=cx(沿 dim3=x)，target[t,1]=cy(沿 dim2=y)
                x = torch.floor(target[t, 0]).long().clamp(0, feat_w - 1)
                y = torch.floor(target[t, 1]).long().clamp(0, feat_h - 1)
                c = target[t, 4].long()

                noobj_mask[b, k, y, x] = 0
                y_true[b, k, y, x, 0] = target[t, 0] % 1
                y_true[b, k, y, x, 1] = target[t, 1] % 1
                y_true[b, k, y, x, 2] = target[t, 2]
                y_true[b, k, y, x, 3] = target[t, 3]
                y_true[b, k, y, x, 4] = 1
                y_true[b, k, y, x, c + 5] = 1
                box_loss_scale[b, k, y, x] = (
                    target[t, 2] * target[t, 3] / feat_h / feat_w
                )

        return y_true, noobj_mask, box_loss_scale

    def get_ignore(
        self,
        bs: int,
        feat_h: int,
        feat_w: int,
        anchors_mask: list[int],
        predict: torch.Tensor,
        targets: list[torch.Tensor],
        noobj_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从模型输出 decode 出预测框坐标，并标记 IoU > 0.5 的背景 cell 为忽略，返回 (noobj_mask, pred_boxes, grid_x, grid_y)。"""
        scaled_anchors = self.anchors[anchors_mask] / (IMG_W / feat_h)
        cx, cy, w, h, grid_x, grid_y = decode_preds(
            predict, scaled_anchors, feat_h, feat_w, self.device
        )

        pred_boxes = torch.cat(
            [
                cx.unsqueeze(-1),
                cy.unsqueeze(-1),
                w.unsqueeze(-1),
                h.unsqueeze(-1),
            ],
            dim=-1,
        )

        for b in range(bs):
            pred_flat = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                anch_ious = compute_iou(targets[b][:, :4], pred_flat)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > 0.5] = 0

        return noobj_mask, pred_boxes, grid_x, grid_y

    def box_giou(
        self,
        b1: torch.Tensor,
        b2: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ) -> torch.Tensor:
        """计算两个框集合的 GIoU，输入为 (bs, N, H, W, 4) 形状的 (cx,cy,w,h) 框，返回同形状 GIoU。"""
        b1_wh_half = b1[..., 2:4] / 2.0
        b1_mins = b1[..., :2] - b1_wh_half
        b1_maxes = b1[..., :2] + b1_wh_half

        b2_xy = b2[..., :2] + torch.stack([grid_x, grid_y], dim=-1)
        b2_wh_half = b2[..., 2:4] / 2.0
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(
            intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes)
        )
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        b1_area = b1[..., 2:4].prod(dim=-1)
        b2_area = b2[..., 2:4].prod(dim=-1)
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(
            enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes)
        )
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        giou = iou - (enclose_area - union_area) / enclose_area
        return giou


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """计算两组框的 IoU，支持广播，box_a(N,4) 与 box_b(M,4) 返回 (N,M)。"""
    n, m = box_a.shape[0], box_b.shape[0]
    if n == 0 or m == 0:
        return torch.zeros(n, m, device=box_a.device, dtype=box_a.dtype)
    box_a = box_a.view(-1, 4)
    box_b = box_b.view(-1, 4)
    a_wh = box_a[:, 2:4].unsqueeze(1)
    b_wh = box_b[:, 2:4].unsqueeze(0)
    a_min = box_a[:, :2].unsqueeze(1) - a_wh / 2
    a_max = box_a[:, :2].unsqueeze(1) + a_wh / 2
    b_min = box_b[:, :2].unsqueeze(0) - b_wh / 2
    b_max = box_b[:, :2].unsqueeze(0) + b_wh / 2

    inter_w = torch.min(a_max[..., 0], b_max[..., 0]) - torch.max(
        a_min[..., 0], b_min[..., 0]
    )
    inter_h = torch.min(a_max[..., 1], b_max[..., 1]) - torch.max(
        a_min[..., 1], b_min[..., 1]
    )
    inter = torch.clamp(inter_w, min=0) * torch.clamp(inter_h, min=0)

    area_a = a_wh[..., 0] * a_wh[..., 1]
    area_b = b_wh[..., 0] * b_wh[..., 1]
    union = area_a + area_b - inter
    iou = inter / union
    return iou


def compute_iou_with_anchors(
    boxes: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """计算目标框与 anchor 的宽高 IoU（以原点为中心构造虚拟框）。

    boxes:    (N, 4) 或 (N, 2)  cx,cy,w,h 或 w,h
    anchors:  (M, 2)  w,h
    返回: (N, M)
    """
    n, m = boxes.shape[0], anchors.shape[0]
    if n == 0 or m == 0:
        return torch.zeros(n, m, device=boxes.device, dtype=boxes.dtype)
    boxes_wh = boxes[:, 2:4] if boxes.shape[-1] >= 4 else boxes[:, :2]
    boxes_zero = torch.cat(
        [
            torch.zeros(n, 2, device=boxes.device, dtype=boxes.dtype),
            boxes_wh,
        ],
        dim=-1,
    )
    anchor_boxes = torch.cat(
        [
            torch.zeros(m, 2, device=boxes.device, dtype=boxes.dtype),
            anchors,
        ],
        dim=-1,
    )
    return compute_iou(boxes_zero, anchor_boxes)


def compute_stride(
    model: nn.Module, input_size: int, device: torch.device
) -> list[int]:
    """通过前向传播，根据输出特征图尺寸反推每层的 stride。"""
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        feature_out = model(dummy_input)
    strides = []
    for feature_map in feature_out:
        if isinstance(feature_map, torch.Tensor) and feature_map.dim() >= 2:
            strides.append(int(input_size / feature_map.shape[2]))
        else:
            print(f"Warning: Unexpected output type: {type(feature_map)}")
    return strides


def focal_loss(
    pred: torch.Tensor,
    targ: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss：在 BCE 基础上乘以调制因子降低易分类样本权重，pred/targ 为 logits 和 0/1 标签。"""
    loss = nn.BCEWithLogitsLoss(reduction="none")(pred, targ)
    predicts = torch.sigmoid(pred)
    p_t = targ * predicts + (1 - targ) * (1 - predicts)
    alpha_factor = targ * alpha + (1 - targ) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss = loss * alpha_factor * modulating_factor

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
