import json
import math
import os
import sys
import time

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nets.yolo import YoloBody, initialParam
from nets.yolo_loss import YOLOv3LOSS
from nets.yolov3_tiny import YOLOv3Tiny
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import set_seed, worker_init_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    with open("config/trainParameter.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(seed=27)
    batch_size = 64
    epochs = 30
    lr = config["lr"]
    train_dataset = YOLODataset(dataset_type="voc")

    # 收敛降lr
    steps = len(train_dataset) / batch_size * epochs
    oversteps = int(steps * 0.8)
    a = steps - oversteps
    min_lr = 1e-5
    gamma = round(math.exp((math.log(min_lr) - math.log(lr)) / a), 4)

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = YOLOv3Tiny(num_classes=20).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    loss_fn = YOLOv3LOSS(device=device, l_loc=1, l_cls=1, l_obj=1, num_classes=20)
    writer_path = "runs"
    writer = SummaryWriter(
        f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}'
    )
    # train
    losses = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                optimizer.zero_grad()
                batch_output = model(batch_x)
                loss_params = loss_fn(predict=batch_output, targets=batch_y)
                loss = loss_params["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (batch + 1)
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    **{"epoch": epoch, "loss": f"{loss.item():.4f}", "lr": lr}
                )
                writer.add_scalars(
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
                if global_step > oversteps:
                    lr_scheduler.step()
                global_step += 1
        losses.append(avg_loss)
        if len(losses) == 1 or losses[-1] < losses[-2]:  # 保存更优的model
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, ".checkpoint.pth")
            os.replace(".checkpoint.pth", "tiny_checkpoint.pth")

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
