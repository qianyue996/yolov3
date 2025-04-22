import numpy as np
import cv2 as cv
import torch

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

class Dynamic_lr():
    def __init__(self):
        super(Dynamic_lr, self).__init__()
        self.losses = []
    def __call__(self, optimizer, lr, loss):
        self.losses.append(loss)

        lr_pool = []
        for param_group in optimizer.param_groups:
            lr_pool.append(param_group['lr'])
        new_lr = np.array(lr_pool).mean()
        try:
            if len(self.losses) == 2:
                if self.losses[-1] > self.losses[-2]:
                    new_lr = lr - lr * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.losses = []
                else:
                    new_lr = lr + lr * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.losses = []
        except Exception as e:
            pass
        return new_lr