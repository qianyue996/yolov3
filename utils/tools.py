import numpy as np
import cv2 as cv
import torch
import os
import shutil
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

class Dynamic_lr():
    def __init__(self):
        super(Dynamic_lr, self).__init__()
        self.losses = []
    def __call__(self, optimizer, lr, loss):
        self.losses.append(loss)
        new_lr = lr
        try:
            if len(self.losses) == 2:
                if self.losses[-1] > self.losses[-2]:
                    new_lr = lr - lr * 0.1
                else:
                    new_lr = lr + lr * 0.1
                new_lr = max(min(new_lr, 1e-2), 1e-6)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.losses = []
            return new_lr
        except Exception as e:
            pass

def nms(boxes, scores, iou_threshold):
    if len(boxes) != len(scores):
        print('boxes and scores length is not equal!')
        return
    
def compute_iou(box_a, box_b):
    '''
    box_a = [x1, y1, x2, y2] shape: (N, 4)
    box_b = [x1, y1, x2, y2] shape: (N, 4)
    '''
    min_xy = torch.min(box_a[:, :2], box_b[:, :2])
    max_xy = torch.max(box_a[:, 2:], box_b[:, 2:])

    inter = max_xy - min_xy
    pass

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def clean_folder(folder_path='runs', keep_last=5):
    if not os.path.exists(folder_path):
        print(f"{folder_path} 不存在")
        return

    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path)]
    subfolders.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # 保留最新 keep_last 个，其余都删
    to_delete = subfolders[keep_last:]

    for folder in to_delete:
        shutil.rmtree(folder)
