"""Darknet-53 主干网络在 Mini-ImageNet100 分类数据集上的预训练脚本。

训练产出纯净的 Backbone 权重字典，可直接被 YoloBody(pretrained=True) 加载。
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
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
from utils import set_seed, worker_init_fn
from utils.augment import augment_image
from utils.config import NORMALIZE_MEAN, NORMALIZE_STD
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
    parser = argparse.ArgumentParser(description="Darknet-53 Backbone 预训练脚本")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/ai_models/mini_imagenet100",
        help="Mini-ImageNet 数据集根目录路径",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="分类类别总数",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="训练与验证输入图像分辨率",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批次大小",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="总训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="初始学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="权重衰减",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader 进程数",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights/backbone",
        help="权重保存目录",
    )
    parser.add_argument(
        "--export-backbone",
        type=str,
        default="model_data/darknet53_backbone_weights.pth",
        help="自动同步导出的最佳 Backbone 权重文件路径（供 YoloBody 直接加载）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的权重文件路径",
    )
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=42)

    save_dir = Path(args.weights_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(args.export_backbone)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 复用 utils.augment 与 utils.transforms 的增强与 letterbox 管道
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
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
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # 5 epoch warmup + Cosine 退火
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-5
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
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_top1 = 0.0
        train_top5 = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
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
                desc=f"Epoch {epoch}/{args.epochs} [Val]",
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
            f"Epoch [{epoch}/{args.epochs}] "
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


if __name__ == "__main__":
    main()
