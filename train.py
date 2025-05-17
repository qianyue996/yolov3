import argparse
import os
import time
import torch
import tqdm
import yaml
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
from pathlib import Path
import platform

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
from utils.torch_utils import load_checkpoint
from utils.yolo_trainning import CustomLR, save_bestmodel, continue_train

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="yolov3-tiny.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/voc.yaml", help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")


if __name__ == "__main__":
    cfg = check_yaml("yolov3-tiny.yaml")
    with open(cfg, encoding="ascii", errors="ignore") as f:
        config = yaml.safe_load(f)

    train_type = "tiny"  # or yolov3
    dataset_type = "voc"
    set_seed(seed=27)
    batch_size = 1
    epochs = 300
    lr = 0.03
    train_dataset = YOLODataset(dataset_type=dataset_type)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    # model = Model(cfg).to(device)
    # load_checkpoint(device, 'models/tiny_weight.pth', model)
    #==================================================#
    #   加载训练
    #==================================================#
    model = continue_train(r"C:\Users\admin\Downloads-h\5.0597_best_58.pt", device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=80, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = YOLOv3LOSS(model=model)
    writer_path = "runs"
    writer = SummaryWriter(f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    start_epoch = 0
    # train
    losses = []
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad()
        total_samples = 0
        total_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                with torch.cuda.amp.autocast():
                    batch_output = model(batch_x)
                    loss_params = loss_fn(batch_output, batch_y)
                    loss = loss_params["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                o_loss = loss_params['original_loss']
                # loss compute
                batch_size = batch_x.shape[0]
                total_loss += o_loss * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                # loss compute
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(**{
                    "ep": epoch,
                    "loss": f"{loss.item():.4f}",
                    "o_loss": f"{avg_loss:.4f}",
                    "lr": lr})
                pbar.write(f"np: {loss_params['np']} | loc: {loss_params['loss_loc']:.4f} | cls: {loss_params['loss_cls']:.4f} | obj: {loss_params['loss_obj']:.4f}")
                writer.add_scalars(
                    "loss",
                    {
                        "loss": loss.item(),
                        "o_loss": avg_loss,
                        "loss_loc": loss_params["loss_loc"],
                        "loss_obj": loss_params["loss_obj"],
                        "loss_cls": loss_params["loss_cls"],
                        "lr": lr,
                    },
                    global_step,
                )
                global_step += 1
        losses.append(avg_loss)
        lr_scheduler.step()
        parameters = {
            'avg_loss': avg_loss,
            'lr': lr,
        }
        writer.add_scalars("parameters", parameters, epoch)
        save_bestmodel(losses, model, optimizer, epoch)
