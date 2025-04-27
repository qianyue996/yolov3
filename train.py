from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
from tqdm import tqdm
import gradio as gr
import threading
from multiprocessing import Process

from nets.yolo import YoloBody, initialParam
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import DynamicLr, set_seed, worker_init_fn
from nets.yolo_loss import YOLOv3LOSS

from config.model_config import yolov3_cfg
from config.dataset_config import dataset_cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class Trainer():
    def __init__(self):
        #
        self.config = get_config()
        #
        set_seed(seed = 27)
        self.batchsize = self.config['batchsize']
        self.lr = self.config['lr']

        self.num_workers = 4
        self.anchors_mask = yolov3_cfg['yolov3']['anchors_mask']
        self.dataset_num_class = dataset_cfg['coco']['num_classes']

    def setup(self):
        self.train_dataset = YOLODataset()
        self.dataloader = DataLoader(dataset=self.train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=self.num_workers,
                                    worker_init_fn=worker_init_fn,
                                    collate_fn=yolo_collate_fn)
        self.model = YoloBody(num_classes = self.dataset_num_class).to(device)
        initialParam(self.model)
        self.model.backbone.load_state_dict(torch.load("models/darknet53_backbone_weights.pth"))
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=5e-4,
                                    weight_decay=5e-4)
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = 5e-4
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                          max_lr=1e-2,
                                                          steps_per_epoch=len(self.dataloader),
                                                          epochs=10,
                                                          pct_start=0.3,
                                                          anneal_strategy='cos',
                                                          div_factor=25,
                                                          final_div_factor=1e4)
        self.loss_fn = YOLOv3LOSS()
        writer_path = 'runs'
        self.writer=SummaryWriter(f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')
        #=======================================================#
        #   尝试读取上次训练进度
        #=======================================================#
        # self.continue_train()
        self.model.train()

    def train(self):
        losses         = []
        global_step    = 0
        for epoch in range(10):
            epoch_loss = 0
            with tqdm(self.dataloader, disable=False) as bar:
                for batch, item in enumerate(bar):
                    batch_x, batch_y = item

                    batch_x = batch_x.to(device)

                    batch_y = [i.to(device) for i in batch_y]
                    
                    self.optimizer.zero_grad()

                    batch_output = self.model(batch_x)

                    loss_params = self.loss_fn(predict=batch_output,
                                                targets=batch_y)
                    
                    loss = loss_params['loss']

                    loss.backward()

                    self.optimizer.step()

                    epoch_loss += loss.item()

                    avg_loss = epoch_loss / (batch + 1)

                    lr = self.optimizer.param_groups[0]['lr']
                            
                    bar.set_postfix(**{'epoch':epoch,
                                    'loss':f'{avg_loss:.4f}',
                                    'lr':lr})
                    
                    self.writer.add_scalars('loss', {'avg_loss':avg_loss,
                                                     'loss_loc':loss_params['loss_loc'],
                                                     'obj_conf':loss_params['obj_conf'],
                                                     'noobj_conf':loss_params['noobj_conf'],
                                                     'loss_cls':loss_params['loss_cls'],
                                                     'lr':lr}, global_step)
                    
                    global_step += 1

                    self.config = get_config()
            losses.append(avg_loss)

            self.save_best_model(epoch=epoch,
                                 losses=losses)

    def save_best_model(self,epoch,losses):
        if len(losses)==1 or losses[-1]<losses[-2]: # 保存更优的model
            checkpoint={
                'model' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'epoch' : epoch
            }
            torch.save(checkpoint, '.checkpoint.pth')
            os.replace('.checkpoint.pth', 'checkpoint.pth')

        EARLY_STOP_PATIENCE = 5   # 早停忍耐度
        if len(losses) >= EARLY_STOP_PATIENCE:
            early_stop = True
            for i in range(1, EARLY_STOP_PATIENCE):
                if losses[-i] < losses[-i-1]:
                    early_stop = False
                    break
            if early_stop:
                print(f'early stop, final loss={losses[-1]}')
                sys.exit()
    
    def continue_train(self, checkpoint=None):
        try:
            checkpoint=torch.load('checkpoint.pth', map_location=device)
            print(f'load model success, start epoch={self.start_epoch}')
        except Exception as e:
            print(e)
            print('load model failed,start from scratch')
            pass

        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch']
            except Exception as e:
                print(e)
                pass

def get_config():
    with open('config/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    trainer=Trainer()

    trainer.setup()
    trainer.train()