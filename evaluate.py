"""YOLOv3 模型独立评估脚本。

使用标准 COCO / VOC 评估指标（Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95）评测已保存的模型权重。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from utils.dataloader import CocoDataset, YOLODataset, yolo_collate_fn
from utils.metrics import EvalResult, evaluate_dataset
from utils.postprocess import _get_model, _load_model, class_names


def print_metrics_table(result: EvalResult) -> None:
    """格式化打印评估指标表格。"""
    header = (
        f"{'Class':<16} {'Targets':<8} {'Precision':<10} {'Recall':<8} "
        f"{'F1':<8} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # 打印各类别指标
    total_targets = sum(cm.num_targets for cm in result.class_metrics)
    for cm in result.class_metrics:
        print(
            f"{cm.class_name:<16} {cm.num_targets:<8d} {cm.precision:<10.4f} "
            f"{cm.recall:<8.4f} {cm.f1:<8.4f} {cm.ap50:<10.4f} {cm.ap50_95:<12.4f}"
        )

    print(separator)
    # 打印所有类别的平均汇总
    print(
        f"{'all':<16} {total_targets:<8d} {result.mp:<10.4f} "
        f"{result.mr:<8.4f} {2*result.mp*result.mr/(result.mp+result.mr+1e-16):<8.4f} "
        f"{result.map50:<10.4f} {result.map50_95:<12.4f}"
    )
    print(separator + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv3 验证与评估工具")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="1000_0.2988.pth",
        help="待评估的模型权重路径（默认 1000_0.2988.pth）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco_val.txt",
        help="验证集文本标签文件路径",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="",
        help="直接使用 COCO JSON 时的标注路径，优先级高于 --data",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="图片根目录，与 --annotation 配合使用",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="评估时的 batch 大小（默认 4）",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="评估时的置信度阈值（默认 0.001，用于完整构建 PR 曲线）",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.6,
        help="评估时的 NMS IoU 阈值（默认 0.6）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 进程数（默认 4）",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU 评估线程数（默认自动使用全部物理核心数）",
    )
    args = parser.parse_args()

    # 配置 CPU 多核并行
    num_threads = args.threads if args.threads is not None else (os.cpu_count() or 4)
    torch.set_num_threads(num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    if not Path(args.checkpoint).exists():
        logger.error(f"权重文件不存在: {args.checkpoint}")
        sys.exit(1)

    logger.info(f"正在加载模型权重: {args.checkpoint}")
    _load_model(args.checkpoint)
    model = _get_model().to(device)

    # 2. 构建验证数据集
    if args.annotation:
        val_dataset = CocoDataset(
            annotation_path=args.annotation, image_root=args.image_root
        )
    elif Path(args.data).exists():
        val_dataset = YOLODataset(labels_path=args.data)
    else:
        logger.error(f"验证集文件不存在: {args.data}")
        sys.exit(1)

    logger.info(f"验证集包含 {len(val_dataset)} 张图片")

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=yolo_collate_fn,
    )

    # 3. 执行评估
    logger.info("开始执行模型评估...")
    eval_result = evaluate_dataset(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    # 4. 打印指标结果
    print_metrics_table(eval_result)


if __name__ == "__main__":
    main()
