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