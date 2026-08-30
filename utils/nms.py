import torch
import torchvision.ops as ops


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_candidates: int = 4096,
) -> torch.Tensor:
    """GPU 优化的多类别非极大值抑制 (Batched NMS)。

    利用向量化分数过滤与 torchvision.ops.batched_nms C++/CUDA 原生算子，
    消除 Python 类别循环与多次 CPU-GPU 阻塞同步。

    Args:
        prediction: (N, 5 + num_classes) 或 (1, N, 5 + num_classes)
                    格式为 [cx, cy, w, h, obj_conf, cls_0, cls_1, ...]
        conf_thres: 置信度阈值 (objectness * class_prob)
        iou_thres: NMS 重叠 IoU 阈值
        max_det: 单图最多输出目标框数量
        max_candidates: 送入 NMS 计算的最高分候选框上限

    Returns:
        (M, 6) 张量，每行格式为 [x1, y1, x2, y2, score, class_id]
    """
    if prediction.ndim == 3:
        prediction = prediction.squeeze(0)

    if prediction.shape[0] == 0:
        return torch.empty((0, 6), dtype=torch.float32, device=prediction.device)

    # 1. 向量化计算每个框的最高类别概率与类别 ID
    obj_conf = prediction[:, 4]  # (N,)
    cls_scores, cls_indices = torch.max(prediction[:, 5:], dim=1)  # (N,)
    scores = obj_conf * cls_scores  # (N,)

    # 2. 向量化快速过滤低置信度框
    mask = scores > conf_thres
    if not mask.any():
        return torch.empty((0, 6), dtype=torch.float32, device=prediction.device)

    boxes_cxcywh = prediction[mask, :4]
    filtered_scores = scores[mask]
    filtered_classes = cls_indices[mask].float()

    # 3. 限制候选框上限，防止低分噪点压垮计算
    if filtered_scores.shape[0] > max_candidates:
        topk_idx = torch.topk(filtered_scores, max_candidates).indices
        boxes_cxcywh = boxes_cxcywh[topk_idx]
        filtered_scores = filtered_scores[topk_idx]
        filtered_classes = filtered_classes[topk_idx]

    # 4. 中心坐标转为左上右下坐标 [cx, cy, w, h] -> [x1, y1, x2, y2]
    half_w = boxes_cxcywh[:, 2] / 2
    half_h = boxes_cxcywh[:, 3] / 2
    boxes_xyxy = torch.stack(
        [
            boxes_cxcywh[:, 0] - half_w,
            boxes_cxcywh[:, 1] - half_h,
            boxes_cxcywh[:, 0] + half_w,
            boxes_cxcywh[:, 1] + half_h,
        ],
        dim=1,
    )

    # 5. 执行 GPU 原生 batched_nms
    keep = ops.batched_nms(
        boxes_xyxy, filtered_scores, filtered_classes.long(), iou_thres
    )
    if keep.shape[0] > max_det:
        keep = keep[:max_det]

    return torch.cat(
        [
            boxes_xyxy[keep],
            filtered_scores[keep].unsqueeze(1),
            filtered_classes[keep].unsqueeze(1),
        ],
        dim=1,
    )
