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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
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
        help="按 epoch 保存 checkpoint 的间隔（默认 1，每个 epoch 结束保存一次；设为 0 关闭）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="按 step 步数保存 checkpoint 的间隔（设置后将自动关闭按 epoch 保存）",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="checkpoint 输出目录（默认 weights）",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="额外保存 avg_loss 最低的模型到 <weights-dir>/best.pth",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 Darknet-53 主干，只训练 FPN + 检测头（需先加载 backbone 预训练权重）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 工作进程数（默认 4，数据加载慢可调大）",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="每隔多少步写一次 TensorBoard 标量（默认 10，降低主进程开销）",
    )
    args = parser.parse_args()

    set_seed(seed=27)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_path = Path(args.weights_dir)
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
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
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
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = YOLOLOSS(model)
    writer_path = "runs"
    writer = SummaryWriter(
        f"{writer_path}/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    # ── 数据流向速查 ─────────────────────────────────────────────────────
    # 1. DataLoader 输出 TransformedBatch
    #       batch_x : ModelInput         (B, 3, 416, 416) float32
    #       batch_y : list[Tensor(N,5)]  xyxy 格式，归一化到 [0,1]（除以 416）
    #
    # 2. 模型前向 RawPredicts
    #       outputs[i] : (B, 3, Hi, Wi, 5+C)  每个 cell 3 个 anchor 的原始 logit
    #       其中 (H0,W0)=(13,13) stride=32 大尺度，(H2,W2)=(52,52) stride=8 小尺度
    #
    # 3. YOLOLOSS 内部
    #       xyxy2xywh   : batch_y → FeatureTargets (cx,cy,w,h) 归一化到 grid
    #       build_targets: FeatureTargets → TargetBuild (y_true, noobj_mask, box_loss_scale)
    #       get_ignore  : RawPredicts → PredDecode (noobj_mask, pred_boxes grid单位)
    #       box_giou    : PredDecode vs y_true → GIoU → loss_loc
    #       各 BCE      : 计算 loss_conf / loss_cls
    #
    # 4. 返回
    #       total_loss : scalar
    #       detail     : {layer0: LayerMetrics, layer1: ..., layer2: ...}
    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        avg_loss = 0
        total_samples = 0
        total_loss = 0

        with tqdm(dataloader) as pbar:
            for _n_batch, item in enumerate(pbar):
                batch_x, batch_y = item  # TransformedBatch
                # pin_memory + non_blocking：H2D 拷贝异步化，不阻塞主进程
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = [i.to(device, non_blocking=True) for i in batch_y]
                outputs = model(batch_x)  # RawPredicts

                loss, detail = loss_fn(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
                if args.save_best and avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model, ".checkpoint.pth")
                    os.replace(".checkpoint.pth", save_path / "best.pth")
                    print(f"  [best] avg_loss={avg_loss:.4f} → {save_path / 'best.pth'}")
                if args.save_every and global_step % args.save_every == 0:
                    step_path = save_path / f"step{global_step}_{avg_loss:.4f}.pth"
                    torch.save(model, ".checkpoint.pth")
                    os.replace(".checkpoint.pth", step_path)
                    print(f"  [step {global_step}] avg_loss={avg_loss:.4f} → {step_path}")

        # 仅在未指定 --save-every 时按 epoch 周期保存
        if not args.save_every and args.save_epoch > 0 and (epoch + 1) % args.save_epoch == 0:
            epoch_path = save_path / f"epoch{epoch}_{avg_loss:.4f}.pth"
            torch.save(model, ".checkpoint.pth")
            os.replace(".checkpoint.pth", epoch_path)
            print(f"[epoch {epoch}] avg_loss={avg_loss:.4f} → {epoch_path}")
            print(f"[epoch {epoch}] avg_loss={avg_loss:.4f} → {epoch_path}")
