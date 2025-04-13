import numpy as np
import cv2 as cv
import torch

def iou(gt, anchors):
    """
    gt: (N, 2) — 每个目标框的 [w, h]
    anchors: (M, 2) — 所有 anchor 的 [w, h]
    
    return: (N, M) — 第 i 行第 j 列是第 i 个 gt 和第 j 个 anchor 的 IOU
    """

    gt = gt.unsqueeze(1)         # (N, 1, 2)
    anchors = anchors.unsqueeze(0)  # (1, M, 2)

    # 计算交集面积
    inter = torch.min(gt, anchors).prod(2)  # (N, M)

    # 计算并集面积
    union = (gt.prod(2) + anchors.prod(2) - inter)  # (N, M)

    # 返回 IOU
    return inter / union  # shape: (N, M)




    pass

def xywh2xyx2(box: list):
    try:
        x,y,w,h = box
        x1,y1 = int(x-w/2),int(y-h/2)
        x2,y2 = int(x+w/2),int(y+h/2)
    except Exception as e:
        print(f'错误：{e}')
    return [x1,y1,x2,y2]

def xyxy2xywh(box: list):
    try:
        x1,y1,x2,y2 = box
        x,y = int((x1+x2)/2),int((y1+y2)/2)
        w,h = int(x2-x1),int(y2-y1)
    except Exception as e:
        print('检查是否x2<x1 / y2<y1')
    return [x,y,w,h]
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持CV2读入的BRG转RGB
#---------------------------------------------------------#
def cvtColor(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
def preprocess_input(image):
    image /= 255.0
    return image