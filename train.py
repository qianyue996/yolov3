"""YOLOv3 训练与主干预训练主脚本。

支持：
1. detect 模式：YOLOv3 / YOLOv3-Tiny 目标检测训练（支持多尺度训练、数据增强、快速验证损失评估与 TensorBoard 监控）。
2. backbone 模式：Darknet-53 分类主干预训练（Mini-ImageNet100 分类、Top-1/Top-5 准确率评估、纯净权重导出）。
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from nets.darknet import darknet53
from nets.yolov3 import YoloBody
from nets.yolov3_tiny import YOLOv3Tiny
from utils import YOLOLOSS, load_classes, set_seed, worker_init_fn
from utils.augment import augment_image
from utils.config import (
    DEFAULT_ANCHORS,
    DEFAULT_ANCHORS_MASK,
    DEFAULT_CLASSES_PATH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TINY_ANCHORS,
    TINY_ANCHORS_MASK,
)
from utils.dataloader import CocoDataset, YOLODataset, yolo_collate_fn
from utils.transforms import letterbox_image


class MiniImageNetDataset(Dataset):
    """Mini-ImageNet CSV 标注格式数据集。"""

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        img_rel_path = str(row["image:FILE"])
        label = int(row["category"])

        img_full_path = self.root_dir / img_rel_path
        image = Image.open(img_full_path).convert("RGB")

        if self.transform is not None:
            tensor_image = cast(torch.Tensor, self.transform(image))
        else:
            tensor_image = cast(torch.Tensor, transforms.ToTensor()(image))

        return tensor_image, label


class DarkNetClassifier(nn.Module):
    """带全局平均池化与分类头的 Darknet-53 图像分类网络。"""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.backbone = darknet53()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out3: 52x52x256, out4: 26x26x512, out5: 13x13x1024 (输入 416x416 时)
        _, _, out5 = self.backbone(x)
        feat = self.avgpool(out5)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[float]:
    """计算 Top-K 准确率 (%)。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return res


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLOv3 训练与主干预训练脚本")
    parser.add_argument(
        "--mode",
        type=str,
        default="detect",
        choices=["detect", "backbone"],
        help="训练模式：detect (YOLO 目标检测训练，默认) 或 backbone (Darknet-53 分类主干预训练)",
    )

    # ===== 目标检测 (detect) 相关参数 =====
    parser.add_argument(
        "--data",
        type=str,
        default="coco_train.txt",
        help="[detect] 标签文本文件路径（由 utils/stratified_sampler 生成）",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="",
        help="[detect] 直接使用 COCO JSON 时指定，优先级高于 --data",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="[detect] 图片根目录，与 --annotation 配合使用",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="",
        help="[detect] 验证集文本标签文件路径（如 data/coco_val_10pct.txt）",
    )
    parser.add_argument(
        "--val-annotation",
        type=str,
        default="",
        help="[detect] 直接使用 COCO JSON 验证集时的标注路径",
    )
    parser.add_argument(
        "--val-image-root",
        type=str,
        default="",
        help="[detect] 验证集图片根目录，与 --val-annotation 配合使用",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="[detect] 冻结 Darknet-53 主干，只训练 FPN + 检测头",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="[detect] 使用 YOLOv3-Tiny 轻量模型结构（2 个检测尺度）",
    )
    parser.add_argument(
        "--img-sizes",
        type=str,
        default="416,448,480,512,544,576",
        help="[detect] 训练输入多尺度列表（以逗号分隔，如 416,448,480；若只传一个尺寸如 416 则固定尺寸）",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=1,
        help="[detect] 按 epoch 保存的间隔（默认 1；设为 0 关闭）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="[detect] 按 step 保存的间隔（设置后将自动关闭按 epoch 保存）",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="[detect] 额外保存最佳模型到 <weights-dir>/best.pth",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="[detect] 开始训练轮数",
    )

    # ===== 主干预训练 (backbone) 相关参数 =====
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/ai_models/mini_imagenet100",
        help="[backbone] Mini-ImageNet 数据集根目录路径",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="[backbone] 分类类别总数",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="[backbone] 分类训练与验证输入图像分辨率（默认 224）",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="[backbone] 权重衰减系数",
    )
    parser.add_argument(
        "--export-backbone",
        type=str,
        default="model_data/darknet53_backbone_weights.pth",
        help="[backbone] 自动同步导出的最佳 Backbone 权重文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="[backbone] 恢复训练的 checkpoint 路径",
    )

    # ===== 通用参数 =====
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="训练 batch 大小（detect 默认 2，backbone 默认 64）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数（detect 默认 120，backbone 默认 100）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="初始学习率（detect 默认 0.01，backbone 默认 0.05）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="预训练权重路径，为 null 时从随机权重开始训练",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="checkpoint 输出目录（detect 默认 weights，backbone 默认 weights/backbone）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 工作进程数",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="关闭训练数据增强",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="TensorBoard 标量写入间隔（步）",
    )
    return parser.parse_args()


def train_detect(args: argparse.Namespace) -> None:
    """YOLOv3 目标检测训练流程。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=27)

    batch_size = args.batch_size if args.batch_size is not None else 2
    epochs = args.epochs if args.epochs is not None else 120
    lr = args.lr if args.lr is not None else 0.01
    weights_dir = args.weights_dir if args.weights_dir is not None else "weights"
    save_path = Path(weights_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(DEFAULT_CLASSES_PATH)
    if args.tiny:
        anchors = TINY_ANCHORS
        anchors_mask = TINY_ANCHORS_MASK
    else:
        anchors = DEFAULT_ANCHORS
        anchors_mask = DEFAULT_ANCHORS_MASK

    try:
        train_sizes = [int(s.strip()) for s in args.img_sizes.split(",") if s.strip()]
    except ValueError:
        train_sizes = [416]
    if not train_sizes:
        train_sizes = [416]

    augment_enabled = not args.no_augment
    train_collate = partial(
        yolo_collate_fn,
        augment=augment_enabled,
        sizes=train_sizes,
    )
    val_collate = partial(
        yolo_collate_fn,
        augment=False,
        sizes=[416],
    )

    # 1. 构建训练数据集
    if args.annotation:
        dataset = CocoDataset(
            annotation_path=args.annotation,
            image_root=args.image_root,
        )
    else:
        dataset = YOLODataset(labels_path=args.data)

    print(
        f"Dataset: {len(dataset)} images "
        f"(from {args.data if not args.annotation else args.annotation})"
        f" | Augment: {augment_enabled} | Scales: {train_sizes}"
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=train_collate,
    )

    # 2. 构建验证数据集（若提供）
    val_dataloader = None
    if args.val_annotation:
        val_dataset = CocoDataset(
            annotation_path=args.val_annotation,
            image_root=args.val_image_root,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )
        print(f"Val Dataset: {len(val_dataset)} images (from {args.val_annotation})")
    elif args.val_data and Path(args.val_data).exists():
        val_dataset = YOLODataset(labels_path=args.val_data)
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=val_collate,
        )
        print(f"Val Dataset: {len(val_dataset)} images (from {args.val_data})")

    # 3. 初始化或加载模型
    if args.checkpoint and args.checkpoint.lower() != "null":
        model = torch.load(
            rf"{args.checkpoint}",
            map_location=device,
            weights_only=False,
        )
    else:
        model_cls = YOLOv3Tiny if args.tiny else YoloBody
        model = model_cls(
            anchors=anchors,
            anchors_mask=anchors_mask,
            class_names=class_names,
        ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone 已冻结，只训练 FPN + 检测头")
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True
        print("Backbone 可训练")

    optimizer = optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        momentum=0.937,
        weight_decay=1e-4,
    )
    loss_fn = YOLOLOSS(model)
    writer_path = "runs"
    writer = SummaryWriter(
        f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )
    start_epoch = args.start_epoch
    global_step = 0
    best_loss = float("inf")

    # 4. 训练主循环
    for epoch in range(start_epoch, epochs):
        model.train()
        avg_loss = 0.0
        total_samples = 0
        total_loss = 0.0

        with tqdm(dataloader) as pbar:
            for _n_batch, item in enumerate(pbar):
                batch_x, batch_y = item  # TransformedBatch
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = [i.to(device, non_blocking=True) for i in batch_y]
                outputs = model(batch_x)

                loss, detail = loss_fn(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                batch_sz = batch_x.shape[0]
                item_loss = loss.item()
                total_loss += item_loss * batch_sz
                total_samples += batch_sz
                avg_loss = total_loss / max(1, total_samples)

                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "step_loss": f"{item_loss:.6f}",
                        "avg_loss": f"{avg_loss:.6f}",
                    }
                )

                if global_step % args.log_every == 0:
                    writer.add_scalars(
                        "yolov3",
                        {"step_loss": item_loss, "avg_loss": avg_loss},
                        global_step,
                    )
                    for layer_idx, layer_detail in detail.items():
                        prefix = f"yolov3/{layer_idx}"
                        writer.add_scalars(
                            prefix,
                            {
                                "loc_loss": layer_detail["loss_loc"],
                                "conf_loss": layer_detail["loss_conf"],
                                "cls_loss": layer_detail["loss_cls"],
                                "center_diff": layer_detail["center_diff"],
                                "wh_diff": layer_detail["wh_diff"],
                                "conf_diff": layer_detail["conf_diff"],
                            },
                            global_step,
                        )

                global_step += 1

                # 按步数保存 checkpoint
                if args.save_every and global_step % args.save_every == 0:
                    step_path = save_path / f"step{global_step}_{avg_loss:.4f}.pth"
                    torch.save(model, ".checkpoint.pth")
                    Path(".checkpoint.pth").replace(step_path)
                    tqdm.write(
                        f"  [step {global_step}] avg_loss={avg_loss:.4f} → {step_path}"
                    )

        # 5. 每个 Epoch 结束后的验证（快速计算 loss 与各项指标，无需耗时的 mAP 评测）
        if val_dataloader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_loc = 0.0
            val_total_conf = 0.0
            val_total_cls = 0.0
            val_samples = 0
            with torch.no_grad():
                val_pbar = tqdm(
                    val_dataloader,
                    desc=f"Epoch {epoch} Val",
                    leave=False,
                )
                for val_item in val_pbar:
                    v_bx, v_by = val_item
                    v_bx = v_bx.to(device, non_blocking=True)
                    v_by = [i.to(device, non_blocking=True) for i in v_by]
                    v_out = model(v_bx)
                    v_loss, v_detail = loss_fn(v_out, v_by)
                    v_bs = v_bx.shape[0]
                    val_total_loss += v_loss.item() * v_bs
                    loc_l = sum(d["loss_loc"] for d in v_detail.values()) / max(
                        1, len(v_detail)
                    )
                    conf_l = sum(d["loss_conf"] for d in v_detail.values()) / max(
                        1, len(v_detail)
                    )
                    cls_l = sum(d["loss_cls"] for d in v_detail.values()) / max(
                        1, len(v_detail)
                    )
                    val_total_loc += loc_l * v_bs
                    val_total_conf += conf_l * v_bs
                    val_total_cls += cls_l * v_bs
                    val_samples += v_bs
                    avg_eval = val_total_loss / max(1, val_samples)
                    val_pbar.set_postfix(
                        {
                            "val_loss": f"{v_loss.item():.4f}",
                            "avg_eval": f"{avg_eval:.4f}",
                        }
                    )

            val_avg_loss = val_total_loss / max(1, val_samples)
            val_avg_loc = val_total_loc / max(1, val_samples)
            val_avg_conf = val_total_conf / max(1, val_samples)
            val_avg_cls = val_total_cls / max(1, val_samples)

            writer.add_scalar("val/loss", val_avg_loss, epoch)
            writer.add_scalar("val/loss_loc", val_avg_loc, epoch)
            writer.add_scalar("val/loss_conf", val_avg_conf, epoch)
            writer.add_scalar("val/loss_cls", val_avg_cls, epoch)
            tqdm.write(
                f"[val epoch {epoch}] avg_loss={val_avg_loss:.4f} "
                f"(loc={val_avg_loc:.4f}, conf={val_avg_conf:.4f}, cls={val_avg_cls:.4f})"
            )

            if args.save_best and val_avg_loss < best_loss:
                best_loss = val_avg_loss
                torch.save(model, ".checkpoint.pth")
                Path(".checkpoint.pth").replace(save_path / "best.pth")
                tqdm.write(
                    f"  [best] avg_val_loss={val_avg_loss:.4f} → {save_path / 'best.pth'}"
                )
        elif args.save_best and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, ".checkpoint.pth")
            Path(".checkpoint.pth").replace(save_path / "best.pth")
            tqdm.write(f"  [best] avg_loss={avg_loss:.4f} → {save_path / 'best.pth'}")

        # 6. 按 Epoch 周期保存 checkpoint
        if (
            not args.save_every
            and args.save_epoch > 0
            and (epoch + 1) % args.save_epoch == 0
        ):
            epoch_path = save_path / f"epoch{epoch}_{avg_loss:.4f}.pth"
            torch.save(model, ".checkpoint.pth")
            Path(".checkpoint.pth").replace(epoch_path)
            tqdm.write(f"[epoch {epoch}] avg_loss={avg_loss:.4f} → {epoch_path}")


def train_backbone(args: argparse.Namespace) -> None:
    """Darknet-53 主干网络分类预训练流程。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=42)

    batch_size = args.batch_size if args.batch_size is not None else 64
    epochs = args.epochs if args.epochs is not None else 100
    lr = args.lr if args.lr is not None else 0.05
    weights_dir = (
        args.weights_dir if args.weights_dir is not None else "weights/backbone"
    )

    save_dir = Path(weights_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_backbone)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 增强与 letterbox 管道
    train_transform = transforms.Compose(
        [
            transforms.Lambda(augment_image),
            transforms.Lambda(
                lambda img: letterbox_image(img, target_size=args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: letterbox_image(img, target_size=args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    train_dataset = MiniImageNetDataset(
        csv_path=train_csv, root_dir=data_dir, transform=train_transform
    )
    val_dataset = MiniImageNetDataset(
        csv_path=val_csv, root_dir=data_dir, transform=val_transform
    )

    logger.info(
        f"数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本"
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # 2. 初始化网络与优化器
    model = DarkNetClassifier(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # 5 epoch warmup + Cosine 退火
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    start_epoch = 0
    best_acc1 = 0.0

    if args.resume:
        logger.info(f"正在从 {args.resume} 恢复模型...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", 0.0)

    writer = SummaryWriter(
        f"runs/backbone_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )

    # 3. 训练主循环
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_top1 = 0.0
        train_top5 = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bs = images.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            train_loss += loss.item() * bs
            train_top1 += acc1 * bs
            train_top5 += acc5 * bs
            total_samples += bs

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "top1": f"{acc1:.2f}%",
                    "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                }
            )

        scheduler.step()

        epoch_train_loss = train_loss / total_samples
        epoch_train_top1 = train_top1 / total_samples
        epoch_train_top5 = train_top5 / total_samples

        writer.add_scalar("train/loss", epoch_train_loss, epoch)
        writer.add_scalar("train/top1", epoch_train_top1, epoch)
        writer.add_scalar("train/top5", epoch_train_top5, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # 4. 验证循环
        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{epochs} [Val]",
                leave=False,
            ):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bs = images.size(0)

                outputs = model(images)
                loss = criterion(outputs, labels)

                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                val_loss += loss.item() * bs
                val_top1 += acc1 * bs
                val_top5 += acc5 * bs
                val_samples += bs

        epoch_val_loss = val_loss / val_samples
        epoch_val_top1 = val_top1 / val_samples
        epoch_val_top5 = val_top5 / val_samples

        writer.add_scalar("val/loss", epoch_val_loss, epoch)
        writer.add_scalar("val/top1", epoch_val_top1, epoch)
        writer.add_scalar("val/top5", epoch_val_top5, epoch)

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} Top1: {epoch_train_top1:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Top1: {epoch_val_top1:.2f}% Top5: {epoch_val_top5:.2f}%"
        )

        # 5. 保存最新模型与最优 Backbone 权重
        checkpoint_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc1": max(best_acc1, epoch_val_top1),
        }
        latest_path = save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint_dict, latest_path)

        if epoch_val_top1 > best_acc1:
            best_acc1 = epoch_val_top1
            best_ckpt_path = save_dir / f"best_top1_{best_acc1:.2f}.pth"
            torch.save(checkpoint_dict, best_ckpt_path)

            # 导出纯净的 Backbone 权重字典，可直接由 YoloBody.backbone.load_state_dict 加载
            backbone_state = model.backbone.state_dict()
            torch.save(backbone_state, export_path)
            logger.info(
                f"🌟 发现更优模型 (Top-1: {best_acc1:.2f}%)，已导出 Backbone 权重至: {export_path}"
            )


def main() -> None:
    """主入口，根据 --mode 路由到对应的训练流程。"""
    args = parse_args()
    if args.mode == "backbone":
        train_backbone(args)
    else:
        train_detect(args)


if __name__ == "__main__":
    main()
