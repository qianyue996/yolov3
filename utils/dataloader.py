from typing import List, Tuple, Any
import torch
import cv2 as cv
import numpy as np
import albumentations as A
from torch.utils.data.dataset import Dataset
import torchvision.datasets.coco as coco
from PIL import ImageDraw, Image
from utils import load_category_config

class_names = load_category_config("config/yolo_conf.yaml")

class YOLODataset(Dataset):
    def __init__(self, root: str, annFile: str):
        """
        Args:
            root (string): 图片存放路径
            annFile (string): 标签文件存放路径
        """
        super().__init__()
        self.dataset = coco.CocoDetection(
            root=root,
            annFile=annFile
        )
        self.transform = A.Compose(
            transforms=[
                A.LongestMaxSize(max_size=416),
                A.PadIfNeeded(min_height=416, min_width=416, border_mode=cv.BORDER_CONSTANT, fill=0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.4711, 0.4475, 0.4080),
                    std=(0.2378, 0.2329, 0.2361),
                    max_pixel_value=255.0
                ),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='coco')
        )

        # 构建原生类名与类别id映射表
        self.id2name = {
            i["id"]: i["name"]
            for i in
            self.dataset.coco.dataset["categories"]
        }

        # 外部类别名称列表
        self.class_name = class_names["coco"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        
        Returns:
            array: 图像的numpy array数组
            bboxes List[List[float]]: 检测框的坐标信息 min_x, min_y, width, height
            labels List[int]: 检测框的类别信息
        """
        image, infos = self.dataset[index]
        info = [
            {
                "bbox": element["bbox"],
                "label": self.class_name.index(self.id2name[element["category_id"]])
            }
            for element in infos
        ]
        transform_result = self.transform(image=np.array(image), bboxes=[i["bbox"] for i in info])
        
        return transform_result["image"], transform_result["bboxes"], [i["label"] for i in info]

def image_show(image: Image.Image, bboxes: List, labels: List):
    image = Image.fromarray(image)
    image_handler = ImageDraw.ImageDraw(image)

    for i, bbox in enumerate(bboxes):
        label_text = f'{class_names["coco"]["class_name"][labels[i]]} {labels[i]}'
        x_min, y_min, width, height = list(map(int, bbox))
        text_x = x_min
        text_y = y_min-15
        image_handler.rectangle(((x_min, y_min), (x_min + width, y_min + height)), outline="red")
        image_handler.text((text_x, text_y), label_text, fill="green")

    img_np_rgb = np.array(image)
    img_np_bgr = cv.cvtColor(img_np_rgb, cv.COLOR_RGB2BGR)
    cv.namedWindow("Image from OpenCV", cv.WINDOW_NORMAL)
    cv.imshow("Image from OpenCV", img_np_bgr)

    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Script continued after closing OpenCV window.")

def yolo_collate_fn(batches: List[Any]):
    total_images = []
    total_labels = []
    for batch in batches:
        image, bboxes, labels = batch
        # image_show(image, bboxes, labels)
        total_images.append(image)
        bboxes = torch.tensor(bboxes)
        labels = torch.tensor(labels).unsqueeze(-1)
        bbox_and_label = torch.cat((bboxes, labels), dim=1)
        total_labels.append(bbox_and_label)
    
    return torch.stack(total_images, dim=0), total_labels
