import json
import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.config import DEFAULT_CLASSES_PATH
from utils.models import RawTargets, TransformedBatch
from utils.transforms import transform

try:
    import yaml

    with open(DEFAULT_CLASSES_PATH) as _f:
        class_names = yaml.safe_load(_f)
except Exception:
    class_names = []


class YOLODataset(Dataset):
    """读取采样工具输出的文本标签文件，格式与 utils/stratified_sampler.py 一致。

    每行格式：
        /path/to/img.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,...

    训练时配合 utils/stratified_sampler.py 生成的采样文件使用，
    也可用于全量标签文件（如 coco_train.txt）。
    """

    def __init__(self, labels_path: str) -> None:
        super().__init__()
        with open(labels_path) as f:
            self.dataset = f.readlines()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Image.Image, RawTargets]:
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
        return image, RawTargets(torch.from_numpy(np_targets))


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

    def __getitem__(self, index: int) -> tuple[Image.Image, RawTargets]:  # type: ignore[override]
        img_id = self.img_ids[int(index)]
        img_path = self.img_id_to_path[img_id]
        image = Image.open(f"{self.image_root}/{img_path}").convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])
        if not anns:
            targets = np.empty((0, 5), dtype=np.float32)
            return image, RawTargets(torch.from_numpy(targets))

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
        return image, RawTargets(torch.from_numpy(np_targets))


def yolo_collate_fn(
    batches: list[Any],
    augment: bool = False,
    sizes: tuple[int, ...] | list[int] = (416,),
) -> TransformedBatch:
    """YOLO DataLoader collate 函数，支持多尺度随机选择与数据增强。

    Args:
        batches: list of (image, raw_targets)
        augment: 是否开启数据增强（翻转、旋转、裁剪、色彩抖动）
        sizes: 可选输入尺度列表（如 [416, 448, 480, 512]）。
               若 augment=True 且 len(sizes) > 1，则本 batch 随机抽取一个尺寸；
               否则固定使用 sizes[0]。

    Returns:
        TransformedBatch(images: (B, 3, S, S), targets: list of (Ni, 5))
    """
    if augment and len(sizes) > 1:
        current_size = random.choice(sizes)  # noqa: S311
    else:
        current_size = sizes[0] if len(sizes) > 0 else 416

    images = []
    targets_list = []
    for batch in batches:
        image, raw = batch
        tb = transform(image, raw, size=current_size, augment=augment)
        images.append(tb.images)
        targets_list.extend(tb.targets)

    return TransformedBatch(torch.stack(images, dim=0), targets_list)
