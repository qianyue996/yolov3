from typing import List
import torch
import torch.nn as nn
import numpy as np
import math


class YOLOLOSS:
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.stride = [8, 16, 32]
        self.anchors = torch.tensor(model.anchors)
        self.anchors_mask = model.anchors_mask
        self.class_name = model.class_name

        self.balance = [4, 1.0, 0.4]
        self.box_ratio = 0.05
        self.obj_ratio = 5
        self.cls_ratio = 1

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def __call__(self, predicts: List[torch.Tensor], targets: List[torch.Tensor]):
        loss = 0
        for l, pred in enumerate(predicts):
            bs = pred.shape[0]
            size_w = pred.shape[2]
            size_h = pred.shape[3]
            anchors_mask = self.anchors_mask[l]

            pred_cls = pred[..., 5:]
            pred_conf = pred[..., 4]
            y_true, noobj_mask, box_loss_scale = self.build_targets(
                l, bs, size_w, size_h, anchors_mask, pred, targets
            )
            noobj_mask, pred_boxes = self.get_ignore(
                l, bs, size_w, size_h, anchors_mask, pred, targets, noobj_mask
            )
            box_loss_scale = 2 - box_loss_scale

            obj_mask = y_true[..., 4] == 1
            n = torch.sum(obj_mask)

            if n != 0:
                giou = self.box_giou(pred_boxes, y_true[..., :4])
                loss_loc = ((1 - giou) * box_loss_scale)[obj_mask].mean()

                pred_cls = pred[..., 5:][obj_mask]
                targ_cls = y_true[..., 5:][obj_mask]
                loss_cls = nn.BCEWithLogitsLoss(reduction="mean")(pred_cls, targ_cls)
                loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

            loss_conf = nn.BCEWithLogitsLoss(reduction="mean")(
                pred_conf, obj_mask.type_as(pred_conf)
            )[noobj_mask.bool() | obj_mask]
            loss += loss_conf * self.balance[l] * self.obj_ratio

        return loss

    def build_targets(
        self, num_layer, bs, size_w, size_h, anchors_mask, predict, targets
    ):
        y_true = torch.zeros_like(predict)
        noobj_mask = torch.ones(
            bs, len(anchors_mask), size_h, size_w, device=self.device
        )
        box_loss_scale = torch.zeros(
            bs, len(anchors_mask), size_w, size_h, device=self.device
        )
        anchors = self.anchors[anchors_mask]
        for b, target in enumerate(targets):
            if len(target) == 0:
                continue
            batch_target = torch.zeros_like(target)
            batch_target[:, [0, 2]] = target[:, [0, 2]] * size_w
            batch_target[:, [1, 3]] = target[:, [1, 3]] * size_h
            batch_target[:, 4] = target[:, 4]
            batch_target = batch_target.cpu()

            iou = compute_iou(batch_target, anchors)
            best_anchors = torch.argmax(iou, dim=-1)

            for t, best_num_anchor in enumerate(best_anchors):
                if best_num_anchor not in anchors_mask:
                    continue
                k = anchors_mask.index(best_num_anchor)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()

                noobj_mask[b, k, i, j] = 0
                y_true[b, k, i, j, 0] = batch_target[t, 0] % 1
                y_true[b, k, i, j, 1] = batch_target[t, 1] % 1
                y_true[b, k, i, j, 2] = batch_target[t, 3]
                y_true[b, k, i, j, 3] = batch_target[t, 4]
                y_true[b, k, i, j, 4] = 1
                y_true[b, k, i, j, c + 5] = 1
                box_loss_scale[b, k, i, j] = (
                    batch_target[t, 2] * batch_target[t, 3] / size_w / size_h
                )

        return y_true, noobj_mask, box_loss_scale

    def get_ignore(
        self, num_layer, bs, size_w, size_h, anchors_mask, predict, targets, noobj_mask
    ):
        x = predict.sigmoid()[..., 0] * 2 - 0.5
        y = predict.sigmoid()[..., 1] * 2 - 0.5
        w = (predict[..., 2].sigmoid() * 2) ** 2
        h = (predict[..., 3].sigmoid() * 2) ** 2
        grid_x = (
            torch.linspace(0, size_w - 1)
            .repeat(size_h, 1)
            .repeat(int(bs * len(anchors_mask)), 1, 1)
            .view(x.shape)
            .type_as(x)
        )
        grid_y = (
            torch.linspace(0, size_h - 1)
            .repeat(size_w, 1)
            .t()
            .repeat(int(bs * len(anchors_mask)), 1, 1)
            .view(y.shape)
            .type_as(x)
        )

        scaled_anchors_l = self.anchors[anchors_mask]
        anchor_w = (
            scaled_anchors_l[:, 0]
            .repeat(bs, 1)
            .repeat(1, 1, size_w * size_h)
            .view(w.shape)
        )
        anchor_h = (
            scaled_anchors_l[:, 1]
            .repeat(bs, 1)
            .repeat(1, 1, size_w * size_h)
            .view(h.shape)
        )

        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(w * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(h * anchor_h, -1)
        pred_boxes = torch.cat(
            [pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1
        )

        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * size_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * size_h
                batch_target = batch_target[:, :4].type_as(x)
                anch_ious = compute_iou(batch_target, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > 0.5] = 0

        return noobj_mask, pred_boxes

    def box_giou(self, b1, b2):
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.0
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.0
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(
            intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes)
        )
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
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


def compute_iou(box_a, box_b):
    boxa_wh = box_a[:, 2:4].unsqueeze(1)
    area_w = torch.min(boxa_wh[..., 0], box_b[..., 0])
    area_h = torch.min(boxa_wh[..., 1], box_b[..., 1])
    inter = area_w * area_h
    boxa_area = boxa_wh[..., 0] * boxa_wh[..., 1]
    boxb_area = box_b[..., 0] * box_b[..., 1]
    union = boxa_area + boxb_area - inter
    iou = inter / union
    return iou


def compute_stride(model, input_size, device):
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        feature_out = model(dummy_input)

    strides = []
    for feature_map in feature_out:
        if isinstance(feature_map, torch.Tensor) and feature_map.dim() >= 2:
            stride = input_size / feature_map.shape[2]
            strides.append(int(stride))
        else:
            print(f"Warning: Unexpected output type or shape: {type(feature_map)}")

    return strides


def focal_loss(pred, targ, alpha=0.25, gamma=1.5, reduction="mean"):
    loss = nn.BCEWithLogitsLoss(reduction="none")(pred, targ)

    predicts = torch.sigmoid(pred)
    p_t = targ * predicts + (1 - targ) * (1 - predicts)
    alpha_factor = targ * alpha + (1 - targ) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= alpha_factor * modulating_factor

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
