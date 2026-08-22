import argparse
import os
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from nets.yolov3 import YoloBody
from utils import YOLOLOSS, load_classes, set_seed, worker_init_fn
from utils.dataloader import CocoDataset, YOLODataset, yolo_collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 训练脚本")
    parser.add_argument(
        "--data",
        type=str,
        default="coco_train.txt",
        help="标签文本文件路径（由 stratified_sampler 生成）",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="",
        help="直接使用 COCO JSON 时指定，优先级高于 --data",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="图片根目录，与 --annotation 配合使用",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="预训练权重路径，为 null 时从随机权重开始训练",
    )
    args = parser.parse_args()

    set_seed(seed=27)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_path = Path("weights")
    os.makedirs(save_path, exist_ok=True)
    class_names = load_classes("data/coco_names.yaml")
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

    # 根据参数选择数据集
    if args.annotation:
        dataset = CocoDataset(
            annotation_path=args.annotation, image_root=args.image_root
        )
    else:
        dataset = YOLODataset(labels_path=args.data)

    print(
        f"Dataset: {len(dataset)} images (from {args.data if not args.annotation else args.annotation})"
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
    )
    # model = YoloBody(
    #     anchors=anchors, anchors_mask=anchors_mask, class_names=class_names
    # ).to(device)
    if args.checkpoint and args.checkpoint.lower() != "null":
        model = torch.load(
            rf"{args.checkpoint}", map_location=device, weights_only=False
        )
    else:
        model = YoloBody(
            anchors=anchors, anchors_mask=anchors_mask, class_names=class_names
        ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = YOLOLOSS(model)
    writer_path = "runs"
    writer = SummaryWriter(
        f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )
    start_epoch = 0
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        avg_loss = 0
        total_samples = 0
        total_loss = 0

        with tqdm(dataloader) as pbar:
            for _n_batch, item in enumerate(pbar):
                batch_x, batch_y = item
                batch_x = batch_x.to(device)
                batch_y = [i.to(device) for i in batch_y]
                outputs = model(batch_x)

                loss = loss_fn(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_sz = batch_x.shape[0]
                item_loss = loss.item()
                total_loss += item_loss * batch_sz
                total_samples += batch_sz
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
