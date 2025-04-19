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
from utils.dataloader import YOLODataset
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

        self.start_epoch=0
        self.losses=[]
        self.checkpoint=None
        self.loss_count = 0

    def setup(self):
        # 加载数据集
        ds=YOLODataset()
        self.dataloader=DataLoader(ds,batch_size=self.batch_size,shuffle=True)
        # 模型初始化
        self.model=YOLOv3().to(self.device)
        self.model.getWeight(self.model)

        # train utils
        self.optimizer=optim.Adam([param for param in self.model.parameters() if param.requires_grad],lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = YOLOv3LOSS()

        # 尝试从上次训练结束点开始
        try:
            self.checkpoint=torch.load('checkpoint.pth', map_location=self.device)
        except Exception as e:
            pass
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.start_epoch = self.checkpoint['epoch'] + 1

        # tensorboard
        self.writer=SummaryWriter(f'runs/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')

        self.model.train()

    def train(self):
        for epoch in range(self.start_epoch,self.epochs):
            all_loss=0
            with tqdm(self.dataloader, disable=False) as bar:
                for batch,item in enumerate(bar):
                    batch_x,batch_y=item
                    batch_x,batch_y=batch_x.to(self.device),[i.to(self.device) for i in batch_y]
                    batch_output=self.model(batch_x)

                    loss = self.loss_fn(predict=batch_output,target=batch_y)

                    loss=loss/len(batch_x)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    all_loss+=loss.item()
                    self.loss_count += 1
                    _loss = all_loss/self.loss_count
                    bar.set_postfix({'epoch':epoch,
                                     'avg_loss:':_loss})
                    self.writer.add_scalar('loss',_loss)
            self.losses.append(_loss)
            self.save_best_model(epoch=epoch)

    def save_best_model(self,epoch):
        if len(self.losses)==1 or self.losses[-1]<self.losses[-2]: # 保存更优的model
            checkpoint={
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(checkpoint,'.checkpoint.pth')
            os.replace('.checkpoint.pth','checkpoint.pth')

        EARLY_STOP_PATIENCE=5   # 早停忍耐度
        if len(self.losses)>=EARLY_STOP_PATIENCE:
            early_stop=True
            for i in range(1,EARLY_STOP_PATIENCE):
                if self.losses[-i]<self.losses[-i-1]:
                    early_stop=False
                    break
                if early_stop:
                    print(f'early stop, final loss={self.losses[-1]}')
                    sys.exit()
    

if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()