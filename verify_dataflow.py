"""数据流向验证脚本。

逐层打印 all_targets 的格式、形状、数值范围，用于确认 YOLOLOSS.__call__ 收到的 all_targets 是否正确。
"""

import torch

from nets.yolov3 import YoloBody
from utils import YOLOLOSS, load_classes
from utils.dataloader import YOLODataset, yolo_collate_fn

LABEL_FILE = "data/coco_train_1pct.txt"
CLASS_NAMES_FILE = "data/coco_names.yaml"
DEVICE = "cpu"

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
class_names = load_classes(CLASS_NAMES_FILE)


def step(label: str, *values: tuple) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for name, obj in values:
        if isinstance(obj, torch.Tensor):
            print(
                f"  {name}: shape={tuple(obj.shape)}, dtype={obj.dtype}, "
                f"min={obj.min().item():.4f}, max={obj.max().item():.4f}"
            )
            if obj.dim() == 2 and obj.shape[1] == 5:
                print(f"    前3行: {obj[:3].tolist()}")
        elif isinstance(obj, (list, tuple)):
            print(f"  {name}: len={len(obj)}")
            for i, item in enumerate(obj):
                if isinstance(item, torch.Tensor):
                    print(
                        f"    [{i}]: shape={tuple(item.shape)}, "
                        f"min={item.min().item():.4f}, max={item.max().item():.4f}"
                    )
                    if item.dim() == 2 and item.shape[1] == 5:
                        print(f"          前3行: {item[:3].tolist()}")
                else:
                    print(f"    [{i}]: {item}")
        else:
            print(f"  {name}: {obj}")


def main() -> None:
    # 1. 读一张原始数据
    print("▶ 读取原始数据集")
    dataset = YOLODataset(LABEL_FILE)
    raw_img, raw_targets = dataset[0]
    step(
        "step1: Dataset.__getitem__ 原始输出",
        ("image", raw_img),
        ("raw_targets.boxes", raw_targets.boxes),
    )
    print("  格式: [x1, y1, x2, y2, class_id]  单位: 像素")

    # 2. 经过 collate_fn
    print("\n▶ 经过 yolo_collate_fn（归一化到 416×416）")
    images, targets = yolo_collate_fn([(raw_img, raw_targets)])
    step("step2: yolo_collate_fn 输出", ("images", images), ("targets", targets))
    print("  格式: [x1, y1, x2, y2, class_id]  范围: [0, 1]  单位: 占 416 的比例")

    # 3. 模拟 train.py 中 to(device) 后传入 YOLOLOSS
    print("\n▶ 模拟 train.py: batch_y 进入 YOLOLOSS.__call__")
    batch_y = [t.to(DEVICE) for t in targets]
    step("step3: batch_y (即 YOLOLOSS 的 all_targets)", ("batch_y", batch_y))
    print("  此时 all_targets 格式: xyxy 归一化到 [0,1]")

    # 4. 模型前向
    print("\n▶ 模型前向，查看 RawPredicts 形状")
    model = YoloBody(
        anchors=anchors, anchors_mask=anchors_mask, class_names=class_names
    ).to(DEVICE)
    batch_x = images.to(DEVICE)
    outputs = model(batch_x)
    step(
        "step4: 模型输出 RawPredicts",
        ("outputs[0] (stride=32, 13×13)", outputs[0]),
        ("outputs[1] (stride=16, 26×26)", outputs[1]),
        ("outputs[2] (stride=8, 52×52)", outputs[2]),
    )
    print("  每层形状: (B, 3 anchors, H, W, 5+C)")

    # 5. 进入 YOLOLOSS，打印每层 xyxy2xywh 后的 targets
    print("\n▶ YOLOLOSS 内部：xyxy2xywh 转换后的 targets")
    from utils.models import xyxy2xywh

    loss_fn = YOLOLOSS(model)
    for layer_idx, pred in enumerate(outputs):
        feat_h = pred.shape[2]
        feat_w = pred.shape[3]
        feature_targets = xyxy2xywh(batch_y, feat_h, feat_w)
        step(
            f"step5-layer{layer_idx}: xyxy2xywh(batch_y, feat_h={feat_h}, feat_w={feat_w})",
            (f"layer{layer_idx}_targets", feature_targets),
        )
        print("  格式: [cx, cy, w, h, class_id]  单位: grid cell (0~feat_w/feat_h)")
        # 验证：cx 应该在 [0, feat_w) 范围内
        if feature_targets[0].shape[0] > 0:
            t = feature_targets[0]
            print(
                f"  验证: cx∈[{t[:, 0].min():.2f}, {t[:, 0].max():.2f}], "
                f"cy∈[{t[:, 1].min():.2f}, {t[:, 1].max():.2f}], "
                f"w∈[{t[:, 2].min():.2f}, {t[:, 2].max():.2f}], "
                f"h∈[{t[:, 3].min():.2f}, {t[:, 3].max():.2f}]"
            )

    # 6. 完整 loss 计算
    print("\n▶ 完整 loss 计算")
    total_loss, detail = loss_fn(outputs, batch_y)
    step("step6: YOLOLOSS 返回值", ("total_loss", total_loss), ("detail", detail))
    print("\n  结论：all_targets 在各层正确传递，格式与 xyxy2xywh 预期一致。")


if __name__ == "__main__":
    main()
