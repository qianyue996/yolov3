import argparse
import os
import time
import torch
import tqdm
import yaml
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
from utils.yolo_trainning import CustomLR, save_best_model, continue_train

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
    batch_size = 8
    epochs = 300
    lr = 0.01
    train_dataset = YOLODataset(dataset_json_path="./voc_train.json")
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = Model(cfg).to(device)
    load_checkpoint(device, 'models/tiny_weight.pth', model)
    model = continue_train(r"0.9945_best_32.pt", device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = YOLOv3LOSS(model=model)
    writer_path = "runs"
    writer = SummaryWriter(f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    start_epoch = 0
    # train
    losses = []
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        avg_loss = 0
        total_samples = 0
        total_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                batch_output = model(batch_x)
                loss_params = loss_fn(batch_output, batch_y)
                loss = loss_params["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # ...
                batch_size = batch_x.shape[0]
                item_loss = loss.item()
                total_loss += item_loss * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                pbar.set_postfix({
                    "epoch": epoch,
                    "step_loss": f"{item_loss:.6f}",
                    "avg_loss": f"{avg_loss:.6f}",})
                pbar.write(f"loc_loss: {loss_params['loc_loss']:.6f} | cls_loss: {loss_params['cls_loss']:.6f} | obj_loss: {loss_params['obj_loss']:.6f}")
                writer.add_scalars(
                    "yolov3",
                    {
                        "step_loss": item_loss,
                        "avg_loss": avg_loss,
                        "loc_loss": loss_params["loc_loss"],
                        "obj_loss": loss_params["obj_loss"],
                        "cls_loss": loss_params["cls_loss"],
                    },
                    global_step,
                )
                global_step += 1
        losses.append(avg_loss)
        save_best_model(losses, model, optimizer, epoch)
