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


from config.yolov3 import CONF
from nets.yolo_copy import YOLOv3
from nets.yolo_loss import YOLOv3LOSS

loss_fn = YOLOv3LOSS()

model = YOLOv3().to(CONF.device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=CONF.device)['model'])
model.eval()

def process(img):
    # 推理（无梯度，加速）
    with torch.no_grad():
        output = model(img)

    result = []

    # 遍历每个特征图输出
    for i in range(len(output)):
        pred = output[i]  # (1, 3 * 85, S, S)
        S = CONF.feature_map[i]
        pred = pred.view(-1, 3, 85, S, S).permute(0, 1, 3, 4, 2)  # (1, 3, S, S, 85)

        best_conf, best_idx = torch.max(torch.sigmoid(pred[..., 4]), dim=1)
        mask = torch.zeros((1, 3, S, S), device=pred.device)
        mask.scatter_(1, best_idx.unsqueeze(1), best_conf.unsqueeze(1))
        obj_mask = mask > 0.5

        anchors = loss_fn.anchors[i]
        noobj_mask, obj_mask = loss_fn.build_target(pred, anchors, thre=0.5)
        # 提取通过置信度阈值的预测框
        filtered_pred = pred[obj_mask].view(-1, 85)

        # 坐标归一化到 [0, 1]
        filtered_pred[:, 0:4] = torch.clamp(filtered_pred[:, 0:4], 0, 1)

        # 分类部分走 sigmoid，得到每类的置信度
        filtered_pred[:, 5:] = torch.sigmoid(filtered_pred[:, 5:])

        # 更新 objectness 分数为：obj_conf * class_conf
        filtered_pred[:, 4] = torch.sigmoid(filtered_pred[:, 4])  # 保守一点先过 sigmoid
        filtered_pred[:, 5:] *= filtered_pred[:, 4:5]

        # 存入列表
        result.append(filtered_pred)

    if result:
        result = torch.cat(result, dim=0)  # 所有特征层的结果拼接
    else:
        result = torch.zeros((0, 85))  # 如果没结果就空 tensor

    return img  # 返回所有置信度 > 0.5 的预测框

def transport(img, to_tensor=True):
    if to_tensor:
        img = cv.resize(img, (CONF.imgsize, CONF.imgsize))
        img = np.transpose(np.array(img / 255.0, dtype=np.float32), (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(torch.float32)
    else:
        img = np.clip(img.squeeze().numpy() * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
    return img

if __name__ == '__main__':
    test_img = r"D:\Python\yolo3-pytorch\img\street.jpg"
    img = cv.imread(test_img)
    img = transport(img, to_tensor=True) # to tensor
    img = process(img) # predict
    img = transport(img, to_tensor=False)
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)
    cv.imshow('Camera', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
