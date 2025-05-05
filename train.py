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

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomLR:
    def __init__(self, optimizer, T_0=10, eta_min=1e-4, step=1):
        self.optimizer = optimizer
        self.steper = step
        self.count = 0
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=2, eta_min=eta_min
        )

    def step(self):
        self.count += 1
        # if self.count % self.steper == 0:
        self.lr_scheduler.step()

    def get_lr(self):
        lr = self.optimizer.param_groups[0]["lr"]
        return lr


def save_bestmodel(model, optimizer, epoch, losses, train_type):
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)

    current_loss = losses[-1]
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    # 只有当非第一轮，且当前为最优时才保存 best
    if epoch > 0 and len(losses) != 1 and current_loss < min(losses[:-1]):
        torch.save(checkpoint, ".checkpoint.pth")
        os.replace(".checkpoint.pth", weights_dir / f"best_{current_loss:.4f}_{epoch}.pth")
    else:
        torch.save(checkpoint, ".checkpoint.pth")
        os.replace(".checkpoint.pth", weights_dir / f"{epoch}_{current_loss:.4f}.pth")



if __name__ == "__main__":
    cfg = check_yaml("yolov3-tiny.yaml")
    with open(cfg, encoding="ascii", errors="ignore") as f:
        config = yaml.safe_load(f)

    train_type = "tiny"  # or yolov3
    dataset_type = "voc"
    continue_train = True
    set_seed(seed=27)
    batch_size = 4
    epochs = 200
    lr = 0.01
    l_loc = 1
    l_cls = 10
    l_obj = 1
    l_noo = 1
    train_dataset = YOLODataset(dataset_type=dataset_type)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = Model(cfg).to(device)
    load_checkpoint(device, 'models/tiny_weight.pth', model)
    for layer in model.model[:13]:
        for param in layer.parameters():
            param.requires_grad = False
    for name, param in model.model.named_parameters():
        print(f"{name}: {param.requires_grad}", end=' ')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = CustomLR(optimizer)
    loss_fn = YOLOv3LOSS(
        model=model,
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    # train
    losses = []
    global_step = 0
    for epoch in range(start_epoch, epochs):
        if epoch + 1 > 30:
            for layer in model.model[:13]:
                for param in layer.parameters():
                    param.requires_grad = True
        model.train()
        total_samples = 0
        epoch_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                optimizer.zero_grad()
                batch_output = model(batch_x)
                loss_params = loss_fn(batch_output, batch_y)
                loss = loss_params["loss"]
                loss.backward()
                optimizer.step()
                # loss compute
                batch_size = batch_x.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
                avg_loss = epoch_loss / total_samples
                # loss compute
                lr = lr_scheduler.get_lr()
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
        writer.add_scalar("avg_loss", avg_loss, epoch)
        save_bestmodel(model, optimizer, epoch, losses, train_type)
