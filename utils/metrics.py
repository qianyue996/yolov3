"""YOLOv3 验证与评估指标计算模块（Precision, Recall, mAP@0.5, mAP@0.5:0.95）。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torchvision.ops as ops
from tqdm import tqdm

from utils.decode import decode_preds
from utils.nms import non_max_suppression


class ClassMetric(NamedTuple):
    """单个类别的评估指标。"""

    class_id: int
    class_name: str
    num_targets: int
    precision: float
    recall: float
    f1: float
    ap50: float
    ap50_95: float


class EvalResult(NamedTuple):
    """验证集整体评估结果。"""

    mp: float  # 平均 Precision
    mr: float  # 平均 Recall
    map50: float  # 平均 mAP@0.5
    map50_95: float  # 平均 mAP@0.5:0.95
    class_metrics: list[ClassMetric]


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """使用全点插值法（COCO/VOC2012 标准）计算 Precision-Recall 曲线下面积 (AP)。

    Args:
        recall: 召回率数组 (N,)
        precision: 精确率数组 (N,)

    Returns:
        AP 标量浮点数
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    class_names: list[str] | None = None,
) -> EvalResult:
    """计算各类别及全类别的 Precision, Recall, mAP@0.5 和 mAP@0.5:0.95。

    Args:
        tp: 预测是否为真阳性的布尔矩阵 (N_preds, 10)，对应 10 个 IoU 阈值 (0.50 ~ 0.95)
        conf: 预测置信度数组 (N_preds,)
        pred_cls: 预测类别数组 (N_preds,)
        target_cls: 真实类别数组 (N_targets,)
        class_names: 类别名称列表

    Returns:
        EvalResult 结构体
    """
    # 按照置信度降序排列
    sort_indices = np.argsort(-conf)
    tp = tp[sort_indices]
    pred_cls = pred_cls[sort_indices]

    unique_classes, target_counts = np.unique(target_cls, return_counts=True)
    num_classes = len(unique_classes)

    class_metrics: list[ClassMetric] = []
    ap50_list: list[float] = []
    ap50_95_list: list[float] = []
    p_list: list[float] = []
    r_list: list[float] = []

    if num_classes == 0:
        return EvalResult(0.0, 0.0, 0.0, 0.0, [])

    for c_idx, c in enumerate(unique_classes):
        c = int(c)
        c_name = class_names[c] if (class_names and c < len(class_names)) else str(c)
        num_gt = int(target_counts[c_idx])

        pred_mask = pred_cls == c
        num_pred = int(np.sum(pred_mask))

        if num_pred == 0 or num_gt == 0:
            class_metrics.append(
                ClassMetric(
                    class_id=c,
                    class_name=c_name,
                    num_targets=num_gt,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    ap50=0.0,
                    ap50_95=0.0,
                )
            )
            continue

        c_tp = tp[pred_mask]
        fpc = (1 - c_tp).cumsum(0)
        tpc = c_tp.cumsum(0)

        # 召回率与精确率
        recall_curve = tpc / num_gt
        precision_curve = tpc / (tpc + fpc)

        p = float(precision_curve[-1, 0])
        r = float(recall_curve[-1, 0])
        f1 = float(2 * p * r / (p + r + 1e-16))

        # 计算 10 个 IoU 阈值下的 AP
        ap_10 = []
        for iou_idx in range(tp.shape[1]):
            ap_10.append(compute_ap(recall_curve[:, iou_idx], precision_curve[:, iou_idx]))

        ap50 = ap_10[0]
        ap50_95 = float(np.mean(ap_10))

        class_metrics.append(
            ClassMetric(
                class_id=c,
                class_name=c_name,
                num_targets=num_gt,
                precision=p,
                recall=r,
                f1=f1,
                ap50=ap50,
                ap50_95=ap50_95,
            )
        )
        ap50_list.append(ap50)
        ap50_95_list.append(ap50_95)
        p_list.append(p)
        r_list.append(r)

    mean_p = float(np.mean(p_list)) if p_list else 0.0
    mean_r = float(np.mean(r_list)) if r_list else 0.0
    mean_ap50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    mean_ap50_95 = float(np.mean(ap50_95_list)) if ap50_95_list else 0.0

    return EvalResult(
        mp=mean_p,
        mr=mean_r,
        map50=mean_ap50,
        map50_95=mean_ap50_95,
        class_metrics=class_metrics,
    )


def evaluate_batch(
    predictions: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    iou_thresholds: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算单个 Batch 中预测框与真实框在 10 个 IoU 阈值下的匹配情况。"""
    batch_tp: list[np.ndarray] = []
    batch_conf: list[float] = []
    batch_pred_cls: list[int] = []
    batch_target_cls: list[int] = []

    for b, pred in enumerate(predictions):
        gt = targets_list[b]  # (N_gt, 5) xyxy, class_id (归一化到 [0,1])
        if len(gt) > 0:
            batch_target_cls.extend(gt[:, 4].long().cpu().tolist())

        if len(pred) == 0:
            continue

        pred_boxes = pred[:, :4]  # (N_pred, 4) xyxy (归一化到 [0, 416])
        pred_scores = pred[:, 4].cpu().tolist()
        pred_classes = pred[:, 5].long().cpu().tolist()

        batch_conf.extend(pred_scores)
        batch_pred_cls.extend(pred_classes)

        if len(gt) == 0:
            batch_tp.append(np.zeros((len(pred), len(iou_thresholds)), dtype=np.uint8))
            continue

        # 将 gt 从 [0, 1] 放大到 416 像素尺度
        gt_boxes = gt[:, :4].to(pred.device) * 416.0
        gt_classes = gt[:, 4].long().to(pred.device)

        # 匹配每个类别的预测与 GT
        tp_matrix = torch.zeros(
            (len(pred), len(iou_thresholds)), dtype=torch.uint8, device=pred.device
        )
        ious = ops.box_iou(pred_boxes, gt_boxes)  # (N_pred, N_gt)

        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            matched_gt = set()
            for p_idx in range(len(pred)):
                p_cls = pred_classes[p_idx]
                best_iou = float(iou_thresh)
                best_gt_idx = -1

                for g_idx in range(len(gt)):
                    if g_idx in matched_gt or gt_classes[g_idx] != p_cls:
                        continue
                    if ious[p_idx, g_idx] > best_iou:
                        best_iou = float(ious[p_idx, g_idx])
                        best_gt_idx = g_idx

                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    tp_matrix[p_idx, iou_idx] = 1

        batch_tp.append(tp_matrix.cpu().numpy())

    tp_arr = np.concatenate(batch_tp, axis=0) if batch_tp else np.empty((0, len(iou_thresholds)), dtype=np.uint8)
    return (
        tp_arr,
        np.array(batch_conf, dtype=np.float32),
        np.array(batch_pred_cls, dtype=np.int64),
        np.array(batch_target_cls, dtype=np.int64),
    )


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
) -> EvalResult:
    """在整个数据集上运行验证并输出完整的 mAP 评测指标。"""
    model.eval()
    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)

    all_tp: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_target_cls: list[np.ndarray] = []

    anchors = torch.tensor(model.anchors, device=device, dtype=torch.float32)
    anchors_mask = model.anchors_mask

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        images, targets = batch  # TransformedBatch
        images = images.to(device, non_blocking=True)
        outputs = model(images)

        # 解码每一层的预测框
        decoded_layers = []
        for i, output in enumerate(outputs):
            _, _, feat_h, feat_w, _ = output.shape
            stride = 416.0 / feat_h
            scaled_anchors = anchors[anchors_mask[i]] / stride
            cx, cy, w, h, _, _ = decode_preds(
                output, scaled_anchors, feat_h, feat_w, device
            )
            # 缩放到 416 像素尺度
            x = cx.unsqueeze(-1) * stride
            y = cy.unsqueeze(-1) * stride
            bw = w.unsqueeze(-1) * stride
            bh = h.unsqueeze(-1) * stride
            conf = output.sigmoid()[..., 4:]

            layer_preds = torch.cat([x, y, bw, bh, conf], dim=-1).view(
                output.shape[0], -1, 5 + model.num_classes
            )
            decoded_layers.append(layer_preds)

        batch_predictions = torch.cat(decoded_layers, dim=1)  # (B, N_all, 5+C)

        # 对 Batch 中每张图片分别执行 NMS
        batch_nms_results: list[torch.Tensor] = []
        for b in range(images.shape[0]):
            pred_b = non_max_suppression(
                batch_predictions[b], conf_thres=conf_thres, iou_thres=iou_thres
            )
            batch_nms_results.append(pred_b)

        tp, conf, pred_cls, target_cls = evaluate_batch(
            batch_nms_results, targets, iou_thresholds
        )

        if len(tp) > 0:
            all_tp.append(tp)
            all_conf.append(conf)
            all_pred_cls.append(pred_cls)
        if len(target_cls) > 0:
            all_target_cls.append(target_cls)

    if not all_target_cls:
        return EvalResult(0.0, 0.0, 0.0, 0.0, [])

    tp_all = np.concatenate(all_tp, axis=0) if all_tp else np.empty((0, 10), dtype=np.uint8)
    conf_all = np.concatenate(all_conf, axis=0) if all_conf else np.empty(0, dtype=np.float32)
    pred_cls_all = np.concatenate(all_pred_cls, axis=0) if all_pred_cls else np.empty(0, dtype=np.int64)
    target_cls_all = np.concatenate(all_target_cls, axis=0)

    return ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all, class_names=class_names)
