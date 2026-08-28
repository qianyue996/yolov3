"""标注与预测可视化调试工具。

模式说明（通过 SHOW_PRED 控制）：
    GT_ONLY    : 只显示 ground truth 框
    PRED_ONLY  : 只显示模型预测框
    BOTH       : 同时显示两种框（GT 绿色、预测红色）

按键：
    ← →  /  Tab  上一张 / 下一张
    s    保存当前图到 output/
    m    切换显示模式
    q    退出
"""

import os
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from utils import load_classes
from utils.postprocess import _load_model, detect
from utils.transforms import image_transform

# ====== 可在下方修改这些常量来切换文件和权重 ======
LABEL_FILE = "data/coco_train_1pct.txt"
CLASS_NAMES_FILE = "data/coco_names.yaml"
WEIGHT_PATH = "1000_0.2988.pth"  # 模型权重路径，为空则只展示 GT
OUTPUT_DIR = "output"

# 显示模式：GT_ONLY / PRED_ONLY / BOTH
SHOW_MODE = "GT_ONLY"

# ====== 以下为内部逻辑，通常无需修改 ======
MODES = ("GT_ONLY", "PRED_ONLY", "BOTH")


def parse_label_line(line: str) -> tuple[str, list[list[float]]]:
    """解析标签文件的一行。

    格式: /path/to/img.jpg x1,y1,x2,y2,class_id  x1,y1,x2,y2,class_id  ...
    """
    parts = line.strip().split(" ")
    img_path = parts[0]
    boxes = [list(map(float, item.split(","))) for item in parts[1:]]
    return img_path, boxes


def class_id_to_color(class_id: int) -> tuple:
    """根据类别 id 生成固定颜色（BGR）。"""
    hue = (int(class_id) * 137) % 256
    return (hue, (hue + 85) % 256, (hue + 170) % 256)


def draw_boxes_and_save(
    items: list[str],
    image: Image.Image,
    output_path: str = "output/verify_boxes.jpg",
) -> np.ndarray:
    """接收 items 列表和图片，绘制所有框后保存到本地，返回 BGR numpy 数组。

    items 格式：
        ['/path/to/img.jpg', 'x1,y1,x2,y2,class_id', ...]
        第一个元素是图片路径（若 image 已提供则忽略），后续每个元素是一条框。
    """
    class_names = load_classes(CLASS_NAMES_FILE)
    boxes: list[list[float]] = []
    for item in items[1:]:
        parts = item.split(",")
        boxes.append([float(p) for p in parts])

    bgr = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        name = (
            class_names[int(class_id)]
            if int(class_id) < len(class_names)
            else str(int(class_id))
        )
        color = class_id_to_color(int(class_id))
        cv.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {int(class_id)}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv.rectangle(
            bgr, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1
        )
        cv.putText(
            bgr,
            label,
            (int(x1), int(y1) - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
        )

    os.makedirs(Path(output_path).parent, exist_ok=True)
    cv.imwrite(output_path, bgr)
    print(f"已保存: {output_path}")
    return bgr


def draw_gt_boxes(
    image: np.ndarray, boxes: list[list[float]], class_names: list[str]
) -> np.ndarray:
    """绘制 ground truth 框，颜色按类别区分。"""
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        name = (
            class_names[int(class_id)]
            if int(class_id) < len(class_names)
            else str(int(class_id))
        )
        color = class_id_to_color(int(class_id))
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {int(class_id)}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv.rectangle(
            image, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1
        )
        cv.putText(
            image,
            label,
            (int(x1), int(y1) - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
        )
    return image


def draw_pred_boxes(
    image: np.ndarray, results: torch.Tensor, class_names: list[str]
) -> np.ndarray:
    """绘制模型预测框，带置信度。"""
    for row in results:
        x1, y1, x2, y2, score, class_id = [float(v) for v in row]
        name = (
            class_names[int(class_id)]
            if int(class_id) < len(class_names)
            else str(int(class_id))
        )
        color = class_id_to_color(int(class_id))
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv.rectangle(
            image, (int(x1), int(y1) - th - 4), (int(x1) + tw, int(y1)), color, -1
        )
        cv.putText(
            image,
            label,
            (int(x1), int(y1) - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
        )
    return image


def run() -> None:
    class_names = load_classes(CLASS_NAMES_FILE)
    print(f"类别数: {len(class_names)}")

    if SHOW_MODE != "GT_ONLY":
        if not WEIGHT_PATH:
            print("WEIGHT_PATH 为空，切换到 GT_ONLY 模式")
            show_mode = "GT_ONLY"
        else:
            _load_model(WEIGHT_PATH)
            print(f"已加载模型: {WEIGHT_PATH}")
            show_mode = SHOW_MODE
    else:
        show_mode = "GT_ONLY"

    with open(LABEL_FILE) as f:
        lines = f.readlines()
    entries = [parse_label_line(line) for line in lines]
    print(f"共 {len(entries)} 张图片  模式={show_mode}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx = 0
    total = len(entries)

    while True:
        img_path, gt_boxes = entries[idx]

        if not os.path.isfile(img_path):
            print(f"[{idx}/{total}] 图片不存在: {img_path}")
            idx = (idx + 1) % total
            continue

        image = cv.imread(img_path)
        if image is None:
            print(f"[{idx}/{total}] 读取失败: {img_path}")
            idx = (idx + 1) % total
            continue

        # 绘制 ground truth
        if show_mode in ("GT_ONLY", "BOTH"):
            image = draw_gt_boxes(image, gt_boxes, class_names)

        # 绘制预测（仅 PRED_ONLY / BOTH 且模型已加载）
        if show_mode in ("PRED_ONLY", "BOTH") and WEIGHT_PATH:
            pil_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(pil_img)
            _, tensor_img = image_transform(pil_img)
            preds = detect(tensor_img.unsqueeze(0))
            image = draw_pred_boxes(image, preds, class_names)

        h, w = image.shape[:2]
        info = (
            f"[{idx}/{total}]  {Path(img_path).name}  "
            f"({w}x{h})  GT={len(gt_boxes)}  mode={show_mode}"
        )
        cv.putText(image, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv.imshow("debug_draw", image)
        key = cv.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = Path(OUTPUT_DIR) / f"{idx:05d}_{Path(img_path).stem}.jpg"
            cv.imwrite(str(out_path), image)
            print(f"已保存: {out_path}")
        elif key == ord("m"):
            cur = MODES.index(show_mode)
            show_mode = MODES[(cur + 1) % len(MODES)]
            print(f"切换到模式: {show_mode}")
        elif key in (ord("\t"), 8):  # Tab / Backspace
            idx = (idx - 1) % total
        elif key == ord(" "):  # Space
            idx = (idx + 1) % total
        elif key == 2555671:  # Left arrow
            idx = (idx - 1) % total
        elif key == 2555923:  # Right arrow
            idx = (idx + 1) % total

    cv.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    run()
