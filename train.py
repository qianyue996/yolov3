from torch.utils.data.dataloader import DataLoader
from torch import optim
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import yaml

from nets.yolo import YoloBody
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import DynamicLr, seed_everything
from nets.yolo_loss import YOLOv3LOSS

with open('config/yolov3.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class Trainer():
    def __init__(self):
        #====================================================#
        #   是否开启自动化模式训练
        #   当达到3次epoch后，解冻backbone，全参训练
        #   默认全参微调为False
        #====================================================#
        self.auto_train = False
        #====================================================#
        #   不同训练模式的batchsize
        #====================================================#
        self.freeze_batch_size = config['training']['freeze_batch_size']
        self.unfreeze_batch_size = config['training']['unfreeze_batch_size']
        #====================================================#
        #   number workers
        #====================================================#
        self.num_workers = config['hardware']['num_workers']

        #====================================================#
        #   seed种子，使得每次独立训练结果一致
        #====================================================#
        seed               = 11
        seed_everything(seed)
        self.batch_size   = 2
        #====================================================#
        #   设备自动选择，优先使用GPU
        #====================================================#
        self.device        = config['hardware']['device']

    def setup(self):
        #====================================================#
        #   加载数据集
        #====================================================#
        self.train_dataset         = YOLODataset(labels_path=config['training']['train_path'],
                                       train=True)
        #====================================================#
        #   数据集加载器
        #====================================================#
        self.dataloader  = DataLoader(self.train_dataset,
                                   batch_size=self.unfreeze_batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=yolo_collate_fn)
        #====================================================#
        #   初始化模型
        #====================================================#
        self.model       = YoloBody(anchors_mask=config['model']['anchors_mask'],
                                    num_classes=config['dataset']['length'],
                                    pretrained=True).to(self.device)
        #====================================================#
        #   初始化优化器
        #====================================================#
        self.optimizer   = optim.Adam(self.model.parameters(),
                                        lr=5e-4,
                                        weight_decay=1e-3)
        #=======================================================#
        #   尝试读取上次训练进度
        #=======================================================#
        checkpoint = None
        try:
            checkpoint=torch.load('checkpoint.pth', map_location=self.device)
        except Exception as e:
            pass
        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
            except Exception as e:
                pass
        #======================================================#
        #   初始化动态学习率
        #====================================================#
        self.lr_scheduler = DynamicLr(self.optimizer, step_size=1, init_lr=5e-4)
        #=======================================================#
        #   初始化损失函数
        #====================================================#
        self.loss_fn = YOLOv3LOSS()
        #=======================================================#
        #   初始化tensorboard
        #=======================================================#
        writer_path = 'runs'
        self.writer=SummaryWriter(f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}')
        #=======================================================#
        #   模型设置为训练模式
        #=======================================================#
        self.model.train()

    def train(self):
        #====================================================#
        #   总loss列表存放，全局步数计数器
        #====================================================#
        losses         = []
        global_step    = 0
        for epoch in range(100):
            #====================================================#
            #   如果epoch超过3次，则解冻backbone，全参训练
            #====================================================#
            if self.auto_train:
                if epoch <= 3:
                    for param in self.model.backbone.parameters():
                        param.requires_grad = False
                    #====================================================#
                    #   使用冻结模式的batchsize加载数据
                    #====================================================#
                    self.dataloader = DataLoader(dataset=self.train_dataset,
                                                batch_size=self.freeze_batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                collate_fn=yolo_collate_fn)
                else:
                    for param in self.model.backbone.parameters():
                        param.requires_grad = True
                    #====================================================#
                    #   使用解冻模式的batchsize加载数据
                    #====================================================#
                    self.dataloader = DataLoader(dataset=self.train_dataset,
                                                batch_size=self.unfreeze_batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                collate_fn=yolo_collate_fn)
            #====================================================#
            #   单个epoch总损失，用于计算epoch内平均损失
            #====================================================#
            epoch_loss = 0
            with tqdm(self.dataloader, disable=False) as bar:
                for batch, item in enumerate(bar):
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
                                    'loss':f'{avg_loss:.4f}',
                                    'lr':lr})
                    
                    self.writer.add_scalars('loss', {'avg_loss':avg_loss,
                                                     'loss_loc':loss_params['loss_loc'],
                                                     'obj_conf':loss_params['obj_conf'],
                                                     'noobj_conf':loss_params['noobj_conf'],
                                                     'loss_cls':loss_params['loss_cls'],
                                                     'lr':lr,
                                                     'loc_l':loss_params['loc_l'],
                                                     'cls_l':loss_params['cls_l'],
                                                     'obj_l':loss_params['obj_l'],
                                                     'noobj_l':loss_params['obj_l']}, global_step)
                    
                    self.lr_scheduler.step(avg_loss)

                    global_step += 1

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
    
if __name__ == '__main__':
    trainer=Trainer()
    trainer.setup()
    trainer.train()