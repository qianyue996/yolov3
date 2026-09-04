from typing import cast

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.augment import apply_augment
from utils.config import (
    IMG_W,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from utils.models import RawTargets, TransformedBatch

_transform_pipeline = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def _preprocess(image: Image.Image) -> torch.Tensor:
    """PIL 图片 → (3, S, S) 归一化张量。

    Compose 的返回值在运行时由 ToTensor 转为 Tensor，
    但 torchvision 类型存根按输入类型推断，故需显式 cast。
    """
    return cast(torch.Tensor, _transform_pipeline(image))


def letterbox(
    image: Image.Image,
    boxes: np.ndarray | None = None,
    target_size: int = IMG_W,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, np.ndarray | None]:
    """等比例缩放图像最长边至 target_size，短边填充 pad_value 至正方形，并同步映射 boxes。

    Args:
        image: PIL 图像
        boxes: (N, 5) 像素绝对坐标 [x1, y1, x2, y2, class_id]，可为 None
        target_size: 目标正方形边长（如 416）
        pad_value: 填充灰度值 (R, G, B)，默认 (114, 114, 114)

    Returns:
        (canvas, mapped_boxes): 填充后的 PIL 图像和画布像素坐标系下的 boxes（若原为 None 则返回 None）
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = Image.new("RGB", (target_size, target_size), pad_value)
    canvas.paste(resized_img, (pad_x, pad_y))

    if boxes is None or boxes.shape[0] == 0:
        return canvas, boxes

    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, target_size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, target_size)
    return canvas, boxes


def letterbox_image(
    image: Image.Image,
    target_size: int = IMG_W,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    """纯图片 letterbox 缩放与填充。"""
    canvas, _ = letterbox(
        image, boxes=None, target_size=target_size, pad_value=pad_value
    )
    return canvas


class TransFormer:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        image: Image.Image,
        raw: RawTargets,
        size: int = IMG_W,
        augment: bool = False,
    ) -> TransformedBatch:
        boxes = raw.boxes.numpy().copy()

        if augment:
            image, boxes = apply_augment(image, boxes)

        canvas, boxes = letterbox(image, boxes, target_size=size)
        tensor_image = _preprocess(canvas)

        if boxes is None or boxes.shape[0] == 0:
            return TransformedBatch(
                tensor_image, [torch.empty((0, 5), dtype=torch.float32)]
            )

        # 归一化到 [0, 1]（除以画布尺寸 size）
        boxes[:, :4] = boxes[:, :4] / float(size)
        return TransformedBatch(
            tensor_image, [torch.from_numpy(boxes.astype(np.float32))]
        )


transform = TransFormer()
