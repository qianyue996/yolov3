from collections import defaultdict
from pathlib import Path
from loguru import logger
import sys
import platform
from tqdm import tqdm
import os
import torchvision.datasets.coco as coco

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils import load_classes

class_name = load_classes("data/coco_names.yaml")

root = r"D:\Python\datasets\coco2014\train2014"

annotation_file = r"D:\Python\datasets\coco2014\annotations\instances_train2014.json"

dataset = coco.CocoDetection(root=root, annFile=annotation_file)

# output
train_name = "coco_train.txt"
val_name = "coco_val.txt"

if __name__ == "__main__":
    name_box_id = defaultdict(list)

    id2name = {i["id"]: i["name"] for i in dataset.coco.dataset["categories"]}

    for i in tqdm(range(len(dataset)), desc="parsing..."):
        id = dataset.ids[i]
        path = dataset.coco.loadImgs(id)[0]["file_name"]
        img_path = os.path.join(root, path)
        bboxes = []
        if len(dataset._load_target(id)) == 0:
            logger.info(f"{img_path} has no bbox")
            continue
        for target in dataset._load_target(id):
            xmin, ymin = target["bbox"][:2]
            xmax = target["bbox"][2] + xmin
            ymax = target["bbox"][3] + ymin
            coco_category_id = target["category_id"]
            label = class_name.index(id2name[coco_category_id])
            insert_data = f"{xmin},{ymin},{xmax},{ymax},{label}"
            bboxes.append(insert_data)
        name_box_id[img_path] = bboxes

    with open(train_name, "w", encoding="utf-8") as f:
        for img_path, bboxes in tqdm(name_box_id.items(), desc="writting..."):
            insert_data = f"{img_path} {' '.join(bboxes)}"
            f.write(insert_data + "\n")
