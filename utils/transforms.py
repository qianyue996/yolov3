from typing import cast

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from utils.config import IMG_H, IMG_W, NORMALIZE_MEAN, NORMALIZE_STD
from utils.models import RawTargets, TransformedBatch

try:
    with open("data/coco_names.yaml") as _f:
        import yaml
        class_names = yaml.safe_load(_f)
except Exception:
    class_names = []

try:
    font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()

_transform_pipeline = transforms.Compose(
    [
        transforms.Resize((IMG_W, IMG_H)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def _preprocess(image: Image.Image) -> torch.Tensor:
    """PIL 图片 → (3,416,416) 归一化张量。

    Compose 的返回值在运行时由 ToTensor 转为 Tensor，
    但 torchvision 类型存根按输入类型推断，故需显式 cast。
    """
    return cast(torch.Tensor, _transform_pipeline(image))


class TransFormer:
    def __init__(self) -> None:
        pass

    def __call__(
        self, image: Image.Image, raw: RawTargets
    ) -> TransformedBatch:
        scaled_factor_w = image.size[0] / IMG_W
        scaled_factor_h = image.size[1] / IMG_H

        tensor_image = _preprocess(image)

        if raw.boxes.shape[0] == 0:
            return TransformedBatch(tensor_image, [torch.empty((0, 5), dtype=torch.float32)])

        targets = raw.boxes.numpy().copy()
        targets[:, [0, 2]] = targets[:, [0, 2]] / scaled_factor_w / IMG_W
        targets[:, [1, 3]] = targets[:, [1, 3]] / scaled_factor_h / IMG_H
        return TransformedBatch(tensor_image, [torch.from_numpy(targets)])


transform = TransFormer()


def image_show(image: Image.Image, targets: np.ndarray) -> None:
    image_handler = ImageDraw.ImageDraw(image)

    for label in targets:
        class_id = int(label[4])
        label_text = f"{class_names[class_id]} {class_id}"
        x_min, y_min, x_max, y_max = list(map(int, label[:4]))
        text_x = x_min
        text_y = y_min - 15
        image_handler.rectangle(((x_min, y_min), (x_max, y_max)), outline="red")
        image_handler.text((text_x, text_y), label_text, fill="green", font=font)

    img_np_rgb = np.array(image)
    img_np_bgr = cv.cvtColor(img_np_rgb, cv.COLOR_RGB2BGR)
    cv.namedWindow("Image from OpenCV", cv.WINDOW_NORMAL)
    cv.imshow("Image from OpenCV", img_np_bgr)

    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Script continued after closing OpenCV window.")


def image_transform(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    """对单张图片进行缩放+归一化，返回 (resized_pil_image, tensor_image)。"""
    resized_image = image.resize((IMG_W, IMG_H), Image.Resampling.BILINEAR)
    to_tensor_image = _preprocess(resized_image)
    return resized_image, to_tensor_image
