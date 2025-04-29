import os
import sys
import time

import torch
import tqdm
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nets.yolo_loss import YOLOv3LOSS
from nets.yolov3 import YOLOv3
from nets.yolov3_tiny import YOLOv3Tiny
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import set_seed, worker_init_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    train_type = "tiny"  # or normal
    set_seed(seed=27)
    batch_size = 4
    epochs = 30
    lr = 0.01
    l_loc = 1
    l_cls = 1
    l_obj = 0.5
    l_noo = 1
    train_dataset = YOLODataset(dataset_type="voc")
    num_classes = 20

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    if train_type == "tiny":
        model = YOLOv3Tiny(num_classes=num_classes).to(device)
    elif train_type == "normal":
        model = YOLOv3(num_classes=num_classes, pretrained=False).to(device)
    else:
        raise ValueError("train_type must be tiny or normal")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    loss_fn = YOLOv3LOSS(
        device=device,
        l_loc=l_loc,
        l_cls=l_cls,
        l_obj=l_obj,
        l_noo=l_noo,
        num_classes=num_classes,
    )
    writer_path = "runs"
    writer = SummaryWriter(
        f'{writer_path}/{time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())}'
    )
    # checkpoint = torch.load("checkpoint.pth", map_location=device)
    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
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
                        "loss_noo": loss_params["loss_noo"],
                        "lr": lr,
                    },
                    global_step,
                )
                global_step += 1

        lr_scheduler.step()
        losses.append(avg_loss)
        if len(losses) == 1 or losses[-1] < losses[-2]:  # 保存更优的model
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
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
