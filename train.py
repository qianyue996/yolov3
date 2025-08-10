import time
import os
from pathlib import Path
import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from nets.yolov3 import YoloBody

from utils import load_category_config, YOLOLOSS, set_seed, worker_init_fn
from utils.dataloader import YOLODataset, yolo_collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    set_seed(seed=27)
    batch_size = 3
    epochs = 120
    lr = 0.01
    save_path = Path("weights")
    os.makedirs(save_path, exist_ok=True)
    class_conf = load_category_config("config/yolo_conf.yaml")
    anchors = [
        [10, 13],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
        [116, 90],
        [156, 198],
        [373, 326],
    ]
    anchors_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    root = r"/mnt/nfs/ai_models/coco2014/train2014"
    annotation_file = (
        r"/mnt/nfs/ai_models/coco2014/annotations/instances_train2014.json"
    )
    dataset = YOLODataset(root=root, annFile=annotation_file)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    model = YoloBody(
        anchors=anchors, anchors_mask=anchors_mask, class_name=class_conf["coco"]
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = YOLOLOSS(model)
    writer_path = "runs"
    writer = SummaryWriter(
        f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )
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
                outputs = model(batch_x)

                loss_all = 0
                for l, output in enumerate(outputs):
                    loss = loss_fn(l, output, batch_y)
                    loss_value_all += loss
                loss = loss_all
                # loss = loss / batch_x.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = batch_x.shape[0]
                item_loss = loss.item()
                total_loss += item_loss * batch_size
                total_samples += batch_size
                avg_loss = total_loss / total_samples
                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "step_loss": f"{item_loss:.6f}",
                        "avg_loss": f"{avg_loss:.6f}",
                    }
                )
                writer.add_scalars(
                    "yolov3",
                    {"step_loss": item_loss, "avg_loss": avg_loss},
                    global_step,
                )
                global_step += 1
                if global_step % 1000 == 0:
                    torch.save(model, ".checkpoint.pth")
                    os.replace(
                        ".checkpoint.pth",
                        save_path / f"{global_step}_{avg_loss:.4f}.pth",
                    )
