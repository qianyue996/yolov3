import json
import os
import sys
import time

import gradio as gr
import numpy as np
import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nets.yolo import YoloBody, initialParam
from nets.yolo_loss import YOLOv3LOSS
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import set_seed, worker_init_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:

    def __init__(self):
        set_seed(seed=27)
        self.batch_size = 12
        self.lr = get_config()["lr"]

        self.train_dataset = YOLODataset()
        self.dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            worker_init_fn=worker_init_fn,
            collate_fn=yolo_collate_fn,
        )
        self.model = YoloBody().to(device)
        initialParam(self.model)
        # self.model.backbone.load_state_dict(torch.load("models/darknet53_backbone_weights.pth"))
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.98
        )

        self.loss_fn = YOLOv3LOSS(
            device=device,
            l_loc=get_config()["l_loc"],
            l_cls=get_config()["l_cls"],
            l_obj=get_config()["l_obj"],
        )
        writer_path = "runs"
        self.writer = SummaryWriter(
            f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}'
        )

    def setup(self):
        # =======================================================#
        #   尝试读取上次训练进度
        # =======================================================#
        self.continue_train()

    def train(self):
        losses = []
        global_step = 0
        for epoch in range(30):
            self.model.train()
            epoch_loss = 0
            # for param in self.model.backbone.parameters(): # 冻结backbone
            #     param.requires_grad = False
            with tqdm(self.dataloader, disable=False) as pbar:
                for batch, item in enumerate(pbar):
                    batch_x, batch_y = item
                    batch_x = batch_x.to(device)
                    batch_y = [i.to(device) for i in batch_y]
                    self.optimizer.zero_grad()
                    batch_output = self.model(batch_x)
                    loss_params = self.loss_fn(predict=batch_output, targets=batch_y)
                    loss = loss_params["loss"]
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    avg_loss = epoch_loss / (batch + 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        **{"epoch": epoch, "loss": f"{loss.item():.4f}", "lr": lr}
                    )
                    self.writer.add_scalars(
                        "loss",
                        {
                            "loss": loss.item(),
                            "loss_loc": loss_params["loss_loc"],
                            "loss_obj": loss_params["loss_obj"],
                            "loss_cls": loss_params["loss_cls"],
                            "lr": lr,
                        },
                        global_step,
                    )
                    # 更新lr
                    self.optimizer.param_groups[0]["lr"] = get_config()["lr"]
                    # 更新loss fn
                    self.loss_fn = YOLOv3LOSS(
                        device=device,
                        l_loc=get_config()["l_loc"],
                        l_cls=get_config()["l_cls"],
                        l_obj=get_config()["l_obj"],
                    )
                    self.lr_scheduler.step()
                    global_step += 1
            losses.append(avg_loss)

            self.save_best_model(epoch=epoch, losses=losses)

    def save_best_model(self, epoch, losses):
        if len(losses) == 1 or losses[-1] < losses[-2]:  # 保存更优的model
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, ".checkpoint.pth")
            os.replace(".checkpoint.pth", "checkpoint.pth")

        EARLY_STOP_PATIENCE = 5  # 早停忍耐度
        if len(losses) >= EARLY_STOP_PATIENCE:
            early_stop = True
            for i in range(1, EARLY_STOP_PATIENCE):
                if losses[-i] < losses[-i - 1]:
                    early_stop = False
                    break
            if early_stop:
                print(f"early stop, final loss={losses[-1]}")
                sys.exit()

    def continue_train(self, checkpoint=None):
        try:
            checkpoint = torch.load("checkpoint.pth", map_location=device)
        except Exception as e:
            print(e)
            print("load model failed,start from scratch")

        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                print(e)


def get_config():
    with open("config/trainParameter.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    trainer = Trainer()

    trainer.setup()
    trainer.train()
