from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm

from config.yolov3 import CONF

# from nets.yolo import YOLOv3
from nets.yolo_copy import YOLOv3
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import Dynamic_lr
from nets.yolo_loss import YOLOv3LOSS


class Eval:
    def __init__(self):
        super().__init__()
        self.device = CONF.device
        self.anchors = CONF.anchors
        self.batch_size = CONF.batchsize
        self.epochs = CONF.epochs
        self.IMG_SIZE = CONF.imgsize
        self.weight_decay = CONF.weight_decay
        self.lr = CONF.learning_rate

    def setup(self):
        # 模型初始化
        self.model = YOLOv3().to(self.device)
        checkpoint = torch.load("checkpoint.pth", map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model"])
            self.start_epoch = checkpoint["epoch"] + 1
        except Exception as e:
            pass

        self.model.eval()

    def eval(self, *args, **kwds):
        pass

    def compute_iou(self, gt_box, anchors):
        pass
