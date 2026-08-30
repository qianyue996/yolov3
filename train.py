"""YOLOv3 训练主脚本。

支持自定义数据集、自动验证集 mAP 评估、断点续训与 TensorBoard 监控。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from nets.yolov3 import YoloBody
from nets.yolov3_tiny import YOLOv3Tiny
from utils import YOLOLOSS, load_classes, set_seed, worker_init_fn
from utils.config import (
    DEFAULT_ANCHORS,
    DEFAULT_ANCHORS_MASK,
    DEFAULT_CLASSES_PATH,
    TINY_ANCHORS,
    TINY_ANCHORS_MASK,
)
from utils.dataloader import CocoDataset, YOLODataset, yolo_collate_fn
from utils.metrics import evaluate_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLOv3 训练脚本")
    parser.add_argument(
        "--data",
        type=str,
        default="coco_train.txt",
        help="标签文本文件路径（由 utils/stratified_sampler 生成）",
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
    parser.add_argument(
        "--val-data",
        type=str,
        default="",
        help="验证集文本标签文件路径（如 data/coco_val_10pct.txt）",
    )
    parser.add_argument(
        "--val-annotation",
        type=str,
        default="",
        help="直接使用 COCO JSON 验证集时的标注路径",
    )
    parser.add_argument(
        "--val-image-root",
        type=str,
        default="",
        help="验证集图片根目录，与 --val-annotation 配合使用",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="每隔多少个 epoch 执行一次完整的 mAP 评测（默认 0 关闭；设为 1 或 N 开启）",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="训练 batch 大小")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--start-epoch", type=int, default=0, help="开始训练轮数")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD 学习率")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="预训练权重路径，为 null 时从随机权重开始训练",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=1,
        help="按 epoch 保存的间隔（默认 1；设为 0 关闭）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="按 step 保存的间隔（设置后将自动关闭按 epoch 保存）",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="checkpoint 输出目录",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="额外保存最佳模型到 <weights-dir>/best.pth",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 Darknet-53 主干，只训练 FPN + 检测头",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="使用 YOLOv3-Tiny 轻量模型结构（2 个检测尺度）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 工作进程数",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="TensorBoard 标量写入间隔（步）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed=27)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_path = Path(args.weights_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(DEFAULT_CLASSES_PATH)
    if args.tiny:
        anchors = TINY_ANCHORS
        anchors_mask = TINY_ANCHORS_MASK
    else:
        anchors = DEFAULT_ANCHORS
        anchors_mask = DEFAULT_ANCHORS_MASK

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
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=yolo_collate_fn,
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
            collate_fn=yolo_collate_fn,
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
            collate_fn=yolo_collate_fn,
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
    best_map = 0.0

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
                    tqdm.write(f"  [step {global_step}] avg_loss={avg_loss:.4f} → {step_path}")

        # 5. 每个 Epoch 结束后的验证与评估
        if val_dataloader is not None:
            model.eval()
            val_total_loss = 0.0
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
                    v_loss, _ = loss_fn(v_out, v_by)
                    v_bs = v_bx.shape[0]
                    val_total_loss += v_loss.item() * v_bs
                    val_samples += v_bs
                    avg_eval = val_total_loss / max(1, val_samples)
                    val_pbar.set_postfix(
                        {
                            "val_loss": f"{v_loss.item():.4f}",
                            "avg_eval": f"{avg_eval:.4f}",
                        }
                    )

            val_avg_loss = val_total_loss / max(1, val_samples)
            writer.add_scalar("val/loss", val_avg_loss, epoch)
            tqdm.write(f"[val epoch {epoch}] avg_eval={val_avg_loss:.4f}")

            if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
                eval_res = evaluate_dataset(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    class_names=class_names,
                    desc=f"Epoch {epoch} mAP Eval",
                )
                writer.add_scalar("val/mAP@0.5", eval_res.map50, epoch)
                writer.add_scalar("val/mAP@0.5:0.95", eval_res.map50_95, epoch)
                writer.add_scalar("val/Precision", eval_res.mp, epoch)
                writer.add_scalar("val/Recall", eval_res.mr, epoch)
                tqdm.write(
                    f"[eval epoch {epoch}] P={eval_res.mp:.4f}, R={eval_res.mr:.4f}, "
                    f"mAP@0.5={eval_res.map50:.4f}, mAP@0.5:0.95={eval_res.map50_95:.4f}"
                )

                if args.save_best and eval_res.map50 > best_map:
                    best_map = eval_res.map50
                    torch.save(model, ".checkpoint.pth")
                    Path(".checkpoint.pth").replace(save_path / "best.pth")
                    tqdm.write(f"  [best] mAP@0.5={eval_res.map50:.4f} → {save_path / 'best.pth'}")
            elif args.save_best and val_avg_loss < best_loss:
                best_loss = val_avg_loss
                torch.save(model, ".checkpoint.pth")
                Path(".checkpoint.pth").replace(save_path / "best.pth")
                tqdm.write(f"  [best] avg_eval={val_avg_loss:.4f} → {save_path / 'best.pth'}")
        elif args.save_best and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model, ".checkpoint.pth")
            Path(".checkpoint.pth").replace(save_path / "best.pth")
            tqdm.write(f"  [best] avg_loss={avg_loss:.4f} → {save_path / 'best.pth'}")

        # 6. 按 Epoch 周期保存 checkpoint
        if not args.save_every and args.save_epoch > 0 and (epoch + 1) % args.save_epoch == 0:
            epoch_path = save_path / f"epoch{epoch}_{avg_loss:.4f}.pth"
            torch.save(model, ".checkpoint.pth")
            Path(".checkpoint.pth").replace(epoch_path)
            tqdm.write(f"[epoch {epoch}] avg_loss={avg_loss:.4f} → {epoch_path}")


if __name__ == "__main__":
    main()
