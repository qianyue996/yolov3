import torch
import cv2 as cv
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
        boxes, ids = [label[:4] for label in labels], [label[4] for label in labels]

        image, boxes = self.augment_data(image, boxes)
        # self.chakan(image,boxes,ids)

        image = np.transpose(np.array(image / 255.0, dtype=np.float32), (2, 0, 1))

        boxes = (np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                          (boxes[:, 1] + boxes[:, 3]) / 2,
                          boxes[:, 2] - boxes[:, 0],
                          boxes[:, 3] - boxes[:, 1]], axis=0).T / self.IMG_SIZE)

        labels = np.concatenate([boxes, np.array(ids)[:, None]], axis=1)

        return image, labels
    def augment_data(self, image, labels, hue=.1, sat=.7, val=.4):

        image_h, image_w = image.shape[:2] # 图片高宽

        boxes = labels # 获取bbox [[x1,y1,x2,y2], ...]

        scale = min(self.IMG_SIZE / image_h, self.IMG_SIZE / image_w)

        h, w = int(image_h * scale), int(image_w * scale)

        x = (self.IMG_SIZE - w) // 2

        y = (self.IMG_SIZE - h) // 2

        image = cvtColor(image) # BGR --> RGB

        image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC) # resize

        new_image = np.full((self.IMG_SIZE, self.IMG_SIZE, 3), (128,128,128), dtype=np.uint8)

        new_image[y:y+h, x:x+w] = image

        image = new_image
        #------------------------------#
        # 随机翻转
        #------------------------------#
        flip = self.rand()<.5
        # flip = True
        if flip: image = cv.flip(image,1)
        #------------------------------#
        # HSV色域变换
        #------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val = cv.split(cv.cvtColor(image, cv.COLOR_RGB2HSV))
        dtype = image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        _x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((_x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(_x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(_x * r[2], 0, 255).astype(dtype)
        image = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val)))
        image = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        #------------------------------#
        # 处理bbox位置大小
        #------------------------------#
        boxes = np.array(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + y
        if flip:
            boxes[:, [2, 0]] = self.IMG_SIZE - boxes[:, [0, 2]]

        return image, boxes
    
    def chakan(self, image, boxes, ids):
        cv.namedWindow('show', cv.WINDOW_NORMAL)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            cv.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), thickness=1)
            cv.putText(image, f'{ids[i]} {CONF.class_name[ids[i]]}',
                       (int(x1), int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
        cv.imshow('show', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
def yolo_collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(boxes).type(torch.FloatTensor) for boxes in bboxes]
    return images, bboxes