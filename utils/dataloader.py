import json
import torch
import os
from tqdm import tqdm
import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np

from config.yolov3 import CONF
from utils.tools import *

class YOLODataset(Dataset):
    def __init__(self, train: bool = True):
        super(YOLODataset, self).__init__()
        #------------------------------#
        train_path = r'coco_train.txt'
        self.IMG_SIZE = CONF.imgsize
        self.smb = [13,26,52]
        self.train = train
        self.targets = []
        #------------------------------#
        with open(train_path,'r',encoding='utf-8')as f:
            for target in f.readlines():
                path = target.strip('\n').split(' ')[0]
                labels = target.strip('\n').split(' ')[1:]
                temp_label = []
                for label in labels:
                    temp_label.append([int(label) for label in label.split(',')])
                self.targets.append((path,temp_label))
    def __len__(self):
        return len(self.targets)
    
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        image, labels = cv.imread(self.targets[index][0]), self.targets[index][1]
        boxes,ids = [label[:4] for label in labels], [label[4] for label in labels]
        # image,boxes,ids = self.augment_data(image,labels)
        image, boxes = self.augment_data(image,labels)
        # self.chakan(image,boxes)

        image = np.transpose(np.array(image / 255.0, dtype=np.float32), (2, 0, 1))

        target = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = np.array(boxes[i]) / self.IMG_SIZE
            id = ids[i]
            x = (x2 + x1) / 2
            y = (y2 + y1) / 2
            w = x2 - x1
            h = y2 - y1
            target.append([x,y,w,h,id])
        target = np.array(target, dtype=np.float32)

        return image,target
    def augment_data(self, image,labels,hue=.1, sat=.7, val=.4):
        #------------------------------#
        # 随机数据增强
        # BGR转RGB
        #------------------------------#
        image = cvtColor(image)
        #------------------------------#
        # 获得图像的高宽与目标高宽
        #------------------------------#
        img_h, img_w = image.shape[:2]
        #------------------------------#
        # 获得预测框
        #------------------------------#
        boxes = labels
        #------------------------------#
        # 处理图片大小
        # 统一到self.IMG_SIZE
        #------------------------------#
        scale = min(self.IMG_SIZE/img_w,self.IMG_SIZE/img_h)
        nw,nh = int(img_w*scale),int(img_h*scale)
        dx = (self.IMG_SIZE-nw) // 2
        dy = (self.IMG_SIZE-nh) // 2
        image = cv.resize(image, (nw,nh), interpolation=cv.INTER_CUBIC)
        new_image = np.full((self.IMG_SIZE, self.IMG_SIZE, 3), (128,128,128), dtype=np.uint8)
        new_image[dy:dy+nh, dx:dx+nw] = image
        image = new_image.copy()
        #------------------------------#
        # 随机翻转
        #------------------------------#
        # flip = self.rand()<.5
        flip = False
        if flip: image = cv.flip(image,1)
        #------------------------------#
        # HSV色域变换
        #------------------------------#
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val = cv.split(cv.cvtColor(image_data, cv.COLOR_RGB2HSV))
        dtype = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val)))
        image_data = cv.cvtColor(image_data, cv.COLOR_HSV2RGB)
        # image = image_data.copy()
        #------------------------------#
        # 处理bbox位置大小
        #------------------------------#
        new_bbox = []
        for box in boxes:
            x1,y1,x2,y2,id = box
            x1,y1,x2,y2 = int(x1*scale+dx),int(y1*scale+dy),int(x2*scale+dx),int(y2*scale+dy)
            new_bbox.append([x1,y1,x2,y2,id])
        boxes,ids = [box[:4] for box in new_bbox], [box[-1] for box in new_bbox]
        if flip:
            new_bbox = []
            for box in boxes:
                x1,y1,x2,y2 = box
                new_bbox.append([self.IMG_SIZE-x1,y1,self.IMG_SIZE-x2,y2])
            boxes = new_bbox
        
        # return image,boxes,ids
        return image, boxes
    
    def chakan(self, image, boxes):
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            cv.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), thickness=2)
        cv.imshow('5', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
def yolo_collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes