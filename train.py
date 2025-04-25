from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import shutil

from config.yolov3 import CONF
# from nets.yolo import YOLOv3
from nets.yolo_copy import YOLOv3
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import Dynamic_lr, clean_folder
from utils.logger import Logger
from nets.yolo_loss import YOLOv3LOSS

class Trainer():
    def __init__(self):
        super().__init__()
        self.device       = CONF.device
        self.anchors      = CONF.anchors
        self.batch_size   = CONF.batchsize
        self.epochs       = CONF.epochs
        self.IMG_SIZE     = CONF.imgsize
        self.weight_decay = CONF.weight_decay
        self.lr           = CONF.learning_rate
        self.train_path   = CONF.coco_train_path
        self.val_path     = CONF.coco_eval_path

        self.losses=[]
        self.start_epoch=0
        self.global_step = 0

    def setup(self):
        # 加载数据集
        train_ds=YOLODataset(labels_path=self.train_path)
        self.dataloader=DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True,
                                   collate_fn=yolo_collate_fn)
        # 模型初始化
        self.model=YOLOv3().to(self.device)
        self.model.getWeight(self.model)

        # train utils
        self.optimizer=optim.Adam(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
        self.lr_scheduler    = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.94)
        self.loss_fn = YOLOv3LOSS()

        # 尝试从上次训练结束点开始
        checkpoint = None
        try:
            checkpoint=torch.load('checkpoint.pth', map_location=self.device)
        except Exception as e:
            pass
        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
            except Exception as e:
                pass

        # tensorboard
        writer_path = 'runs'
        clean_folder(folder_path=writer_path, keep_last=3)
        self.writer=SummaryWriter(f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

        self.model.train()

    def train(self):
        for epoch in range(self.start_epoch,self.epochs):
            epoch_loss=0
            with tqdm(self.dataloader, disable=False) as bar:
                for batch,item in enumerate(bar):
                    batch_x, batch_y = item
                    batch_x = batch_x.to(self.device)
                    batch_y = [i.to(self.device) for i in batch_y]
                    
                    batch_output = self.model(batch_x)

                    loss_params = self.loss_fn(predict=batch_output,
                                                targets=batch_y)
                    
                    loss = loss_params['loss'] / self.batch_size

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss+=loss.item()
                    avg_loss = epoch_loss/(batch + 1)

                    for params in self.optimizer.param_groups:
                        lr = params['lr']
                            
                    bar.set_postfix({'epoch':epoch,
                                    'loss':avg_loss,
                                    'lr':lr})
                    
                    self.writer.add_scalars('loss', {'Avg_loss':avg_loss,
                                                     'loss_loc':loss_params['loss_loc'],
                                                     'obj_conf':loss_params['obj_conf'],
                                                     'noobj_conf':loss_params['noobj_conf'],
                                                     'loss_cls':loss_params['loss_cls']}, self.global_step)
                    self.lr_scheduler.step()
                    self.global_step += 1
            self.losses.append(avg_loss)
            self.save_best_model(epoch=epoch)

    def save_best_model(self,epoch):
        if len(self.losses)==1 or self.losses[-1]<self.losses[-2]: # 保存更优的model
            checkpoint={
                'model' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'epoch' : epoch
            }
            torch.save(checkpoint,'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')

        EARLY_STOP_PATIENCE = 5   # 早停忍耐度
        if len(self.losses) >= EARLY_STOP_PATIENCE:
            early_stop = True
            for i in range(1, EARLY_STOP_PATIENCE):
                if self.losses[-i] < self.losses[-i-1]:
                    early_stop = False
                    break
                if early_stop:
                    print(f'early stop, final loss={self.losses[-1]}')
                    sys.exit()
    
if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()