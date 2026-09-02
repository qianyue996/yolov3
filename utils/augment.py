"""数据增强模块：几何变换、随机裁剪与颜色抖动。

同时支持纯图片变换与 (PIL 图片, 像素标注框 (N, 5)) 级同步变换。
"""

from __future__ import annotations

import math
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# 颜色抖动变换器
_color_jitter = transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05,
)


def random_flip_h(
    image: Image.Image, boxes: np.ndarray | None = None
) -> tuple[Image.Image, np.ndarray | None]:
    """水平翻转 (p=0.5)。"""
    w, _ = image.size
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if boxes is None or boxes.shape[0] == 0:
        return image, boxes

    boxes = boxes.copy()
    x1 = boxes[:, 0].copy()
    x2 = boxes[:, 2].copy()
    boxes[:, 0] = w - x2
    boxes[:, 2] = w - x1
    return image, boxes


def random_flip_v(
    image: Image.Image, boxes: np.ndarray | None = None
) -> tuple[Image.Image, np.ndarray | None]:
    """垂直翻转 (p=0.5)。"""
    _, h = image.size
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if boxes is None or boxes.shape[0] == 0:
        return image, boxes

    boxes = boxes.copy()
    y1 = boxes[:, 1].copy()
    y2 = boxes[:, 3].copy()
    boxes[:, 1] = h - y2
    boxes[:, 3] = h - y1
    return image, boxes


def random_rotate90(
    image: Image.Image,
    boxes: np.ndarray | None = None,
    k: int | None = None,
) -> tuple[Image.Image, np.ndarray | None]:
    """90° 整数倍逆时针旋转 (k ∈ {1, 2, 3})。"""
    if k is None:
        k = random.choice([1, 2, 3])  # noqa: S311

    w, h = image.size

    if k == 1:
        # 逆时针 90° (新图宽高: h, w)
        image = image.transpose(Image.Transpose.ROTATE_90)
        if boxes is not None and boxes.shape[0] > 0:
            boxes = boxes.copy()
            x1, y1, x2, y2 = (
                boxes[:, 0].copy(),
                boxes[:, 1].copy(),
                boxes[:, 2].copy(),
                boxes[:, 3].copy(),
            )
            boxes[:, 0] = y1
            boxes[:, 1] = w - x2
            boxes[:, 2] = y2
            boxes[:, 3] = w - x1
    elif k == 2:
        # 180° (新图宽高: w, h)
        image = image.transpose(Image.Transpose.ROTATE_180)
        if boxes is not None and boxes.shape[0] > 0:
            boxes = boxes.copy()
            x1, y1, x2, y2 = (
                boxes[:, 0].copy(),
                boxes[:, 1].copy(),
                boxes[:, 2].copy(),
                boxes[:, 3].copy(),
            )
            boxes[:, 0] = w - x2
            boxes[:, 1] = h - y2
            boxes[:, 2] = w - x1
            boxes[:, 3] = h - y1
    elif k == 3:
        # 逆时针 270° (顺时针 90°，新图宽高: h, w)
        image = image.transpose(Image.Transpose.ROTATE_270)
        if boxes is not None and boxes.shape[0] > 0:
            boxes = boxes.copy()
            x1, y1, x2, y2 = (
                boxes[:, 0].copy(),
                boxes[:, 1].copy(),
                boxes[:, 2].copy(),
                boxes[:, 3].copy(),
            )
            boxes[:, 0] = h - y2
            boxes[:, 1] = x1
            boxes[:, 2] = h - y1
            boxes[:, 3] = x2

    return image, boxes


def random_crop(
    image: Image.Image,
    boxes: np.ndarray | None = None,
    min_area_ratio: float = 0.5,
) -> tuple[Image.Image, np.ndarray | None]:
    """随机裁剪 (面积占比 min_area_ratio~100%)。若有标注框则保证至少保留 1 个有效框。"""
    w, h = image.size
    orig_area = w * h

    for _ in range(50):
        target_area = random.uniform(min_area_ratio, 1.0) * orig_area  # noqa: S311
        aspect_ratio = random.uniform(0.5, 2.0)  # noqa: S311
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

        crop_w = max(16, min(crop_w, w))
        crop_h = max(16, min(crop_h, h))

        x1 = 0 if w == crop_w else random.randint(0, w - crop_w)  # noqa: S311
        y1 = 0 if h == crop_h else random.randint(0, h - crop_h)  # noqa: S311
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped_img = image.crop((x1, y1, x2, y2))

        # 纯图片裁剪（如分类任务）
        if boxes is None or boxes.shape[0] == 0:
            return cropped_img, boxes

        bx1 = np.clip(boxes[:, 0] - x1, 0, crop_w)
        by1 = np.clip(boxes[:, 1] - y1, 0, crop_h)
        bx2 = np.clip(boxes[:, 2] - x1, 0, crop_w)
        by2 = np.clip(boxes[:, 3] - y1, 0, crop_h)

        bw = bx2 - bx1
        bh = by2 - by1
        orig_box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-6
        retained_ratio = (bw * bh) / orig_box_area

        # 过滤退化框与面积残留过小的框 (保留比例需 > 20% 且边长 > 2px)
        valid = (bw > 2) & (bh > 2) & (retained_ratio > 0.2)

        if np.any(valid):
            new_boxes = np.column_stack(
                [bx1[valid], by1[valid], bx2[valid], by2[valid], boxes[valid, 4]]
            )
            return cropped_img, new_boxes.astype(np.float32)

    # 50 次尝试均未成功保留有效框，回退原图
    return image, boxes


def random_color_jitter(image: Image.Image) -> Image.Image:
    """对 PIL 图像应用随机亮度、对比度、饱和度与色调抖动。"""
    return _color_jitter(image)


def apply_augment(
    image: Image.Image, boxes: np.ndarray | None = None
) -> tuple[Image.Image, np.ndarray | None]:
    """按序应用几何与色彩增强链。"""
    # 1. 随机裁剪 (50% 概率)
    if random.random() < 0.5:  # noqa: S311
        image, boxes = random_crop(image, boxes)

    # 2. 随机 90° 旋转 (50% 概率)
    if random.random() < 0.5:  # noqa: S311
        image, boxes = random_rotate90(image, boxes)

    # 3. 随机水平翻转 (50% 概率)
    if random.random() < 0.5:  # noqa: S311
        image, boxes = random_flip_h(image, boxes)

    # 4. 随机垂直翻转 (50% 概率)
    if random.random() < 0.5:  # noqa: S311
        image, boxes = random_flip_v(image, boxes)

    # 5. 随机色彩抖动 (50% 概率)
    if random.random() < 0.5:  # noqa: S311
        image = random_color_jitter(image)

    return image, boxes


def augment_image(image: Image.Image) -> Image.Image:
    """纯图像数据增强（用于 Backbone 分类等无标注框任务）。"""
    aug_img, _ = apply_augment(image, boxes=None)
    return aug_img
