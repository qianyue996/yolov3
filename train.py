import time
import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.yolo import Model

from utils.ComputeLoss import YOLOLOSS
from utils.dataloader import YOLODataset, yolo_collate_fn
from utils.tools import set_seed, worker_init_fn
from utils.yolo_trainning import CustomLR, save_best_model, continue_train

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    set_seed(seed=27)
    batch_size = 8
    epochs = 300
    lr = 0.001
    train_dataset = YOLODataset()
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = Model("models/yolov3.yaml").to(device)
    # load_checkpoint(device, 'models/tiny_weight.pth', model)
    # model = continue_train(r"0.9945_best_32.pt", device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = YOLOLOSS(model)
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

        with tqdm(dataloader) as pbar:
            for n_batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                batch_output = model(batch_x)
                loss_params = loss_fn(batch_output, batch_y)
                loss = loss_params["loss"] / batch_x.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = batch_x.shape[0]
                item_loss = loss.item()
                total_loss += item_loss * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                pbar.set_postfix({
                    "epoch": epoch,
                    "step_loss": f"{item_loss:.6f}",
                    "avg_loss": f"{avg_loss:.6f}",})
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
