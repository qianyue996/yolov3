import os
import time
import torch
import tqdm
import yaml
from torch import optim
from torch.cuda.amp import GradScaler, autocast
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


if __name__ == "__main__":
    cfg = check_yaml("yolov3-tiny.yaml")
    with open(cfg, encoding="ascii", errors="ignore") as f:
        config = yaml.safe_load(f)

    train_type = "tiny"  # or yolov3
    dataset_type = "voc"
    set_seed(seed=27)
    batch_size = 64
    epochs = 100
    lr = 0.0005
    l_loc = 0.05
    l_cls = 1
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
    # load_checkpoint(device, 'models/tiny_weight.pth', model)
    for name, param in model.model.named_parameters():
        print(f"{name}: {param.requires_grad}", end=' ')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # lr_scheduler = CustomLR(optimizer, warm_up=(lr, 0.01, 0), T_max=30, eta_min=1e-4)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)
    scaler = GradScaler()
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
    #==================================================#
    #   加载训练
    #==================================================#
    continue_train('tiny_weight.pth', model, optimizer)
    start_epoch = 0
    # train
    losses = []
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        total_samples = 0
        epoch_loss = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                optimizer.zero_grad()
                with autocast():
                    batch_output = model(batch_x)
                    loss_params = loss_fn(batch_output, batch_y)
                    loss = loss_params["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                o_loss = loss_params['original_loss']
                # loss compute
                batch_size = batch_x.size(0)
                epoch_loss += o_loss.item() * batch_size
                total_samples += batch_size
                avg_loss = epoch_loss / total_samples
                # loss compute
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(**{
                    "ep": epoch,
                    "loss": f"{loss.item():.4f}",
                    "o_loss": f"{avg_loss:.4f}",
                    "lr": lr})
                pbar.write(f"np: {loss_params['np']} | obj: {loss_params['loss_obj'].item():.4f} | noo: {loss_params['loss_noo'].item():.4f}")
                writer.add_scalars(
                    "loss",
                    {
                        "loss": loss.item(),
                        "o_loss": avg_loss,
                        "loss_loc": loss_params["loss_loc"],
                        "loss_obj": loss_params["loss_obj"],
                        "loss_cls": loss_params["loss_cls"],
                        "loss_noo": loss_params["loss_noo"],
                        "lr": lr,
                    },
                    global_step,
                )
                global_step += 1

        losses.append(avg_loss)

        # lr_scheduler.step()
        parameters = {
            'avg_loss': avg_loss,
            'lr': lr,
        }
        writer.add_scalars("parameters", parameters, epoch)
        save_bestmodel(losses, model, optimizer, epoch)
