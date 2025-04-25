import torch
import cv2 as cv
from torch.utils.data.dataset import Dataset
import numpy as np
import tqdm
import yaml

from utils.tools import *

with open('config/yolov3.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class YOLODataset(Dataset):
    def __init__(self, labels_path = '', train=False, val=False):
        super(YOLODataset, self).__init__()
        #------------------------------#
        self.IMG_SIZE = 416
        self.class_names = config['dataset']['class_names']

        self.train = train
        self.val = val

        self.all_labels = []
        #===========================================#
        #   读取标签文件
        #===========================================#
        with open(labels_path,'r',encoding='utf-8')as f:

            with tqdm.tqdm(f.readlines(), desc='读取标签文件...') as bar:

                for item in bar:
                    #===========================================#
                    #   读取图片路径
                    #===========================================#
                    img_path = item.strip('\n').split(' ')[0]
                    #===========================================#
                    #   读取坐标
                    #===========================================#
                    labels = item.strip('\n').split(' ')[1:]

                    single_label = []

                    for label in labels:

                        single_label.append([int(label) for label in label.split(',')])

                    self.all_labels.append((img_path, single_label))

    def __len__(self):
        return len(self.all_labels)
    
    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        #===========================================#
        #   读取图片文件转为numpy格式
        #   labels 为bbox的标签
        #   labels = [[x1,y1,x2,y2,id], ...]
        #===========================================#
        image  = cv.imread(self.all_labels[index][0])
        labels = self.all_labels[index][1]
        #===========================================#
        #   分离bbox和id
        #===========================================#
        boxes  = np.array([label[:4] for label in labels])
        ids    = np.array([label[4] for label in labels])
        #===========================================#
        #   训练和验证使用不同的模式
        #===========================================#
        if self.train:
            image, boxes = self.augment_data(image, boxes, aug=self.train)
        elif self.val:
            image, boxes = self.augment_data(image, boxes)
        else:
            print('请选择train或val')
            return
        # self.chakan(image,boxes,ids)
        #===========================================#
        #   构建目标数据:
        #       - boxes坐标归一化并转换为xywh格式
        #       - 加上类别id
        #   格式为: [[x, y, w, h, id], ...]
        #===========================================#
        labels = self.buildTarget(boxes, ids)

        return image, labels
    def augment_data(self, image, labels, hue=.1, sat=.7, val=.4, aug=False):
        #===========================================#
        #   配置基础参数
        #===========================================#
        image_h, image_w = image.shape[:2] # 图片高宽

        boxes = labels # 获取bbox [[x1,y1,x2,y2], ...]

        scale = min(self.IMG_SIZE / image_h, self.IMG_SIZE / image_w)

        h, w = int(image_h * scale), int(image_w * scale)

        x = (self.IMG_SIZE - w) // 2

        y = (self.IMG_SIZE - h) // 2
        #===========================================#
        #   图片预处理BGE2RGB
        #===========================================#
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #===========================================#
        #   图片缩放到目标大小
        #===========================================#
        image = cv.resize(image, (w, h), interpolation=cv.INTER_CUBIC)
        #===========================================#
        #   填充图片到灰度背景
        #===========================================#
        full_grey = np.full((self.IMG_SIZE, self.IMG_SIZE, 3), (128,128,128), dtype=np.uint8)
        full_grey[y:y+h, x:x+w] = image
        image = full_grey
        #===========================================#
        #   判断是否要数据增强
        #===========================================#
        if aug:
            #===========================================#
            #   开始数据增强
            #===========================================#
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
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + y
        if flip:
            boxes[:, [2, 0]] = self.IMG_SIZE - boxes[:, [0, 2]]

        return image, boxes
    
    def buildTarget(self, boxes, ids):

        x = (boxes[:, 0] + boxes[:, 2]) / 2
        y = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        bboxes = np.stack([x, y, w, h], axis=0).T.astype(np.float32)
        labels = np.concatenate([bboxes, ids[:, None]], axis=1).astype(np.float32)

        return labels

    def chakan(self, image, boxes, ids):
        cv.namedWindow('show', cv.WINDOW_NORMAL)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            cv.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), thickness=1)
            cv.putText(image, f'{ids[i]} {self.class_names[ids[i]]}',
                       (int(x1), int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
        cv.imshow('show', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
def yolo_collate_fn(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F).transpose(2, 0, 1))
        # box
        box[:, :4] = box[:, :4] / 416.0
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.float32)
    bboxes = [torch.from_numpy(boxes).type(torch.float32) for boxes in bboxes]
    return images, bboxes