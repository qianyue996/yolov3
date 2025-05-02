import os
import sys
import time
from pathlib import Path
import platform

import torch
import tqdm
import yaml
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import Model
from utils.general import check_yaml

from utils.ComputeLoss import YOLOv3LOSS
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import set_seed, worker_init_fn

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    cfg = check_yaml("yolov3-tiny.yaml")
    with open(cfg, encoding="ascii", errors="ignore") as f:
        config = yaml.safe_load(f)

    train_type = "tiny"  # or yolov3
    dataset_type = "voc"
    continue_train = False
    set_seed(seed=27)
    batch_size = 4
    epochs = 300
    lr = 0.01
    l_loc = 1
    l_cls = 1
    l_obj = 1
    l_noo = 1
    train_dataset = YOLODataset(dataset_type=dataset_type)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = Model(cfg).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    loss_fn = YOLOv3LOSS(
        device=device,
        l_loc=l_loc,
        l_cls=l_cls,
        l_obj=l_obj,
        l_noo=l_noo,
    )
    writer_path = "runs"
    writer = SummaryWriter(f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    start_epoch = 0
    if continue_train:
        checkpoint = torch.load(f"{train_type}_weight.pth", map_location=device)
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # start_epoch = checkpoint["epoch"] + 1
    # train
    losses = []
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                optimizer.zero_grad()
                batch_output = model(batch_x)
                loss_params = loss_fn(model, predict=batch_output, targets=batch_y)
                loss = loss_params["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (batch + 1)
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(**{"epoch": epoch, "loss": f"{loss.item():.4f}", "lr": lr})
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
            os.replace(".checkpoint.pth", f"{train_type}_weight.pth")

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
