from collections import defaultdict
import json
import tqdm
import os

with open('config/datasetParameter.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

class_name = config['coco']['class_name']

instances_train_path = r'D:\Python\datasets\coco2014\annotations\instances_train2014.json'
instances_val_path = r'D:\Python\datasets\coco2014\annotations\instances_val2014.json'

image_train_path = r'D:\Python\datasets\coco2014\train2014'
image_val_path = r'D:\Python\datasets\coco2014\val2014'

output_train_path = 'coco_train.txt'
output_val_path = 'coco_val.txt'

if __name__ == '__main__':
    name_box_id = defaultdict(list)

    with open(instances_train_path, 'r', encoding='utf-8') as f:
        print("\nloading train annotation...\n")
        train_data = json.load(f)
    id2name = {item["id"]: item["name"] for item in train_data["categories"]}
    for data in tqdm.tqdm(train_data["annotations"], desc="processing... "):
        image_id = data["image_id"]
        full_path = os.path.join(image_train_path, f"COCO_train2014_{image_id:012d}.jpg")
        id = class_name.index(id2name[data['category_id']])
        name_box_id[full_path].append([data["bbox"], id])
    with open(output_train_path, 'w', encoding='utf-8') as f:
        for key in tqdm.tqdm(name_box_id.keys(), desc="writing... "):
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])
                box_info = f" {x_min},{y_min},{x_max},{y_max},{int(info[1])}"
                f.write(box_info)
            f.write("\n")

    name_box_id = defaultdict(list)

    with open(instances_val_path, 'r', encoding='utf-8') as f:
        print("\nloading val annotation...\n")
        val_data = json.load(f)
    id2name = {item["id"]: item["name"] for item in val_data["categories"]}
    for data in tqdm.tqdm(val_data["annotations"], desc="processing... "):
        image_id = data["image_id"]
        full_path = os.path.join(image_val_path, f"COCO_val2014_{image_id:012d}.jpg")
        id = class_name.index(id2name[data['category_id']])
        name_box_id[full_path].append([data["bbox"], id])
    with open(output_val_path, 'w', encoding='utf-8') as f:
        for key in tqdm.tqdm(name_box_id.keys(), desc="writing... "):
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])
                box_info = f" {x_min},{y_min},{x_max},{y_max},{int(info[1])}"
                f.write(box_info)
            f.write("\n")
