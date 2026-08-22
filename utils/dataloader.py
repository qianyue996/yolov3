import json
from typing import Any

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data.dataset import Dataset

from utils import load_classes

class_names = load_classes("data/coco_names.yaml")

try:
    font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()

img_w = 416
img_h = 416


class YOLODataset(Dataset):
    """读取采样工具输出的文本标签文件，格式与 label_util/coco_util.py 一致。

    每行格式：
        /path/to/img.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,...

    训练时配合 label_util/stratified_sampler.py 生成的采样文件使用，
    也可用于全量标签文件（如 coco_train.txt）。
    """

    def __init__(self, labels_path: str) -> None:
        super().__init__()
        with open(labels_path) as f:
            self.dataset = f.readlines()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, np.ndarray]:
        items = self.dataset[index].strip().split(" ")
        image = Image.open(items[0]).convert("RGB")

        labels = []
        for item in items[1:]:
            parts = item.split(",")
            labels.append(list(map(float, parts)))

        np_targets = (
            np.array(labels, dtype=np.float32)
            if labels
            else np.empty((0, 5), dtype=np.float32)
        )
        return image, np_targets


class CocoDataset(Dataset):
    """直接读取 COCO JSON 标注的训练数据集，无需预先导出为文本文件。"""

    def __init__(self, annotation_path: str, image_root: str) -> None:
        super().__init__()
        with open(annotation_path) as f:
            coco_data = json.load(f)

        self.image_root = image_root
        self.img_id_to_path = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }
        self.img_id_to_anns: dict[int, list] = {}
        for ann in coco_data["annotations"]:
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
        self.img_ids = list(self.img_id_to_path.keys())

        self.cat_id_to_label = {}
        for cat in coco_data["categories"]:
            self.cat_id_to_label[cat["id"]] = cat["name"]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> tuple[Image.Image, np.ndarray]:
        img_id = self.img_ids[index]
        img_path = self.img_id_to_path[img_id]
        image = Image.open(f"{self.image_root}/{img_path}").convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])
        if not anns:
            targets = np.empty((0, 5), dtype=np.float32)
            return image, targets

        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            label_name = self.cat_id_to_label.get(
                ann["category_id"], ann["category_id"]
            )
            label_idx = (
                class_names.index(label_name)
                if label_name in class_names
                else ann["category_id"] - 1
            )
            labels.append([x, y, x + w, y + h, float(label_idx)])

        np_targets = np.array(labels, dtype=np.float32)
        return image, np_targets


class TransFormer:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_w, img_h)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4711, 0.4475, 0.4080), std=(0.2378, 0.2329, 0.2361)
                ),
            ]
        )

    def __call__(
        self, image: Image.Image, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_factor_w = image.size[0] / img_w
        scaled_factor_h = image.size[1] / img_h

        image = self.transform(image)

        if targets.shape[0] == 0:
            return image, torch.empty((0, 5), dtype=torch.float32)

        targets = targets.copy()
        targets[:, [0, 2]] = targets[:, [0, 2]] / scaled_factor_w
        targets[:, [1, 3]] = targets[:, [1, 3]] / scaled_factor_h
        targets[:, [0, 2]] = targets[:, [0, 2]] / img_w
        targets[:, [1, 3]] = targets[:, [1, 3]] / img_h

        return image, torch.from_numpy(targets)


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


def yolo_collate_fn(batches: list[Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images = []
    labels = []
    for batch in batches:
        image, label = batch
        image, label = transform(image, label)
        images.append(image)
        labels.append(label)

    return torch.stack(images, dim=0), labels
