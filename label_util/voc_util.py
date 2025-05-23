import json
import os
from collections import defaultdict
import xml.etree.ElementTree as ET
import random
import yaml

import tqdm

with open('config/datasets.yaml', encoding="ascii", errors="ignore")as f:
    cfg = yaml.safe_load(f)
class_names = cfg['voc']['class_name']

# 划分训练集和验证集
split_ratio = 0.8

annotation_path_2007 = r"D:\Python\datasets\voc07+12\VOCdevkit/VOC2007/Annotations"
annotation_path_2012 = r"D:\Python\datasets\voc07+12\VOCdevkit/VOC2012/Annotations"

output_train_path = 'voc_train.txt'
output_val_path = 'voc_val.txt'

if __name__ == "__main__":
    name_box_id = defaultdict(list)

    for annotation_path in [annotation_path_2007, annotation_path_2012]:
        for i, _id in enumerate(tqdm.tqdm(os.listdir(annotation_path))):
            xml_path = os.path.join(annotation_path, _id)
            img_path = os.path.join(annotation_path.replace('Annotations', 'JPEGImages'), _id.split(".")[0] + ".jpg")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for _object in root.findall("object"):
                name = _object.find("name").text
                real_id = class_names.index(name)
                bndbox = _object.find("bndbox")
                xmin = bndbox.find("xmin").text
                ymin = bndbox.find("ymin").text
                xmax = bndbox.find("xmax").text
                ymax = bndbox.find("ymax").text
                bndbox = [xmin, ymin, xmax, ymax]
                name_box_id[img_path].append([bndbox, real_id])

    items = list(name_box_id.items())
    random.shuffle(items)
    split_index = int(len(items) * split_ratio)
    train_box_id = defaultdict(list, items[:split_index])
    val_box_id = defaultdict(list, items[split_index:])

    with open(output_train_path, "w", encoding="utf-8") as f:
        for key in tqdm.tqdm(train_box_id.keys()):
            f.write(key)
            box_infos = train_box_id[key]
            for info in box_infos:
                x_min = info[0][0]
                y_min = info[0][1]
                x_max = info[0][2]
                y_max = info[0][3]
                box_info = f" {x_min},{y_min},{x_max},{y_max},{info[1]}"
                f.write(box_info)
            f.write("\n")

    with open(output_val_path, "w", encoding="utf-8") as f:
        for key in tqdm.tqdm(val_box_id.keys()):
            f.write(key)
            box_infos = val_box_id[key]
            for info in box_infos:
                x_min = info[0][0]
                y_min = info[0][1]
                x_max = info[0][2]
                y_max = info[0][3]
                box_info = f" {x_min},{y_min},{x_max},{y_max},{info[1]}"
                f.write(box_info)
            f.write("\n")
