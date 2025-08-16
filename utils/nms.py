import time
import torch
from torchvision.ops import nms
from loguru import logger


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45
):
    scores = prediction[:, 4]
    class_scores = prediction[:, 5:]

    bboxes = torch.zeros_like(prediction[:, :4])
    bboxes[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    bboxes[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    bboxes[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    bboxes[:, 3] = prediction[:, 1] + prediction[:, 3] / 2

    final_scores = scores.unsqueeze(1) * class_scores

    num_classes = final_scores.shape[1]

    # 存储最终结果的列表
    final_detections = []

    # 对每个类别进行循环
    for i in range(num_classes):
        # 提取当前类别的分数
        class_scores = final_scores[:, i]

        # 根据分数阈值筛选
        high_score_indices = torch.where(class_scores > conf_thres)[0]
        
        # 如果该类别没有高分预测，则跳过
        if high_score_indices.shape[0] == 0:
            continue
        
        # 筛选出当前类别的边界框和分数
        filtered_boxes = bboxes[high_score_indices]
        filtered_scores = class_scores[high_score_indices]
        
        # 应用 NMS
        keep_indices = nms(filtered_boxes, filtered_scores, iou_thres)
        
        # 获取 NMS 后的最终检测结果
        final_boxes_per_class = filtered_boxes[keep_indices]
        final_scores_per_class = filtered_scores[keep_indices]
        
        # 将结果（边界框、分数、类别）存储起来
        # 你可能还需要添加类别标签
        class_labels = torch.full((len(final_scores_per_class),), i, dtype=torch.float32)
        final_detections.append(torch.cat([final_boxes_per_class, final_scores_per_class.unsqueeze(1), class_labels.unsqueeze(1)], dim=1).squeeze())

    print(f"最终检测到的物体总数：{len(final_detections)}")
    return final_detections

