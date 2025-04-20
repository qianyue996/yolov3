import json
import os
from collections import defaultdict
from tqdm import tqdm
import cv2 as cv
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import torch
import torchvision


from config.yolov3 import CONF
from nets.yolo_copy import YOLOv3
from nets.yolo_loss import YOLOv3LOSS

loss_fn = YOLOv3LOSS()

model = YOLOv3().to(CONF.device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=CONF.device)['model'])
model.eval()

def draw(img, boxes, scores, labels):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def process(img, input):
    with torch.no_grad():
        output = model(input)

    all_boxes, all_scores, all_labels = [], [], []

    for i in range(len(output)):
        S = CONF.feature_map[i]

        prediction = output[i].squeeze().view(3, 85, S, S).permute(0, 2, 3, 1)
        anchors = torch.tensor(CONF.anchors[i], device=CONF.device)

        prediction[..., 4] = torch.sigmoid(prediction[..., 4])
        mask = prediction[..., 4] > 0.5

        x = prediction[mask][:, 0].sigmoid() * CONF.feature_map[i] * CONF.net_scaled[i]
        y = prediction[mask][:, 1].sigmoid() * CONF.feature_map[i] * CONF.net_scaled[i]
        w = torch.exp(prediction[mask][:, 2] * anchors[:, 0].view(-1, 1))
        h = torch.exp(prediction[mask][:, 3] * anchors[:, 1].view(-1, 1))
        c = prediction[mask][:, 4]
        _cls = prediction[mask][:, 5:].sigmoid()

        scores = c.unsqueeze(-1) * _cls

        boxes = 

        box_x1y1 = bx_by - bw_bh / 2
        box_x2y2 = bx_by + bw_bh / 2
        boxes = torch.cat([box_x1y1, box_x2y2], dim=-1)  # [3, S, S, 4]

        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])
        scores = obj_conf.unsqueeze(-1) * cls_conf  # [3, S, S, num_classes]

        # reshape everything to 1D
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, 80)

        # 选出所有分数大于阈值的 box + 类别
        score_thresh = 0.3
        for cls_id in range(80):
            cls_scores = scores[:, cls_id]
            keep = cls_scores > score_thresh
            if keep.sum() == 0:
                continue
            cls_boxes = boxes[keep]
            cls_scores = cls_scores[keep]
            cls_labels = torch.full((cls_scores.shape[0],), cls_id, dtype=torch.int64, device=CONF.device)

            all_boxes.append(cls_boxes)
            all_scores.append(cls_scores)
            all_labels.append(cls_labels)

    # 拼接所有层输出
    if not all_boxes:
        return
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # NMS 按类别分别处理（或用 batched_nms）
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # draw
    draw(img, boxes, scores, labels)

def transport(img, to_tensor=True):
    if to_tensor:
        img = cv.resize(img, (CONF.imgsize, CONF.imgsize))
        img = np.transpose(np.array(img / 255.0, dtype=np.float32), (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(torch.float32).to(CONF.device)
    return img

if __name__ == '__main__':
    test_img = r"D:\Python\yolo3-pytorch\img\street.jpg"
    img = cv.imread(test_img)
    input = transport(img, to_tensor=True) # to tensor
    process(img, input) # predict
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)
    cv.imshow('Camera', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
