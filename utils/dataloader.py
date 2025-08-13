from typing import List, Tuple, Any
import torch
import cv2 as cv
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import copy
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset

from utils import load_classes

class_names = load_classes("data/coco_names.yaml")

try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    # 如果找不到 Arial，可以尝试其他字体或使用默认字体
    font = ImageFont.load_default()

img_w = 416
img_h = 416


class YOLODataset(Dataset):
    def __init__(self, labels_path: str):
        """
        Args:
            root (string): 图片存放路径
            annFile (string): 标签文件存放路径
        """
        super().__init__()
        with open(labels_path, "r") as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
            image (Image.Image): 图片
            targets (List): 检测框的类别信息 x_min, y_min, x_max, y_max, label
        """
        items = self.dataset[index].strip().split(" ")
        image = Image.open(items[0]).convert("RGB")

        labels = []
        for item in items[1:]:
            bbox_and_label = []
            item = item.split(",")
            bbox_and_label.extend(list(map(float, item)))
            labels.append(bbox_and_label)

        np_targets = np.array(labels)

        return image, np_targets


class TransFormer:
    def __init__(self) -> None:
        self.transform = T.Compose(
            [
                T.Resize((img_w, img_h)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4711, 0.4475, 0.4080), std=(0.2378, 0.2329, 0.2361)
                ),
            ]
        )

    def __call__(self, image: Image.Image, targets: np.ndarray):
        scaled_factor_w = image.size[0] / img_w
        scaled_factor_h = image.size[1] / img_h

        image = self.transform(image)

        targets[:, [0, 2]] = targets[:, [0, 2]] / scaled_factor_w
        targets[:, [1, 3]] = targets[:, [1, 3]] / scaled_factor_h
        targets = targets[:, [0, 2]] / img_w
        targets = targets[:, [1, 3]] / img_h

        return image, torch.from_numpy(targets)


transform = TransFormer()


def image_show(image: Image.Image, targets: np.ndarray):
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


def yolo_collate_fn(batches: List[Any]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    images = []
    labels = []  # 检测框的坐标信息 min_x, min_y, x_max, y_max, label
    for batch in batches:
        image, label = batch
        # image_show(image, label)
        image, label = transform(image, label)
        images.append(image)
        labels.append(label)

    return torch.stack(images, dim=0), labels
