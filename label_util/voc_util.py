import json
import os
import xml.etree.ElementTree as ET
import random
import yaml
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main(cfg, out_train, out_val):
    big_labels_data = []
    name2id = {n: i for i, n in enumerate(cfg['voc']['class_name'])}

    for annotation_path in [cfg['voc']['annotation_path_2007'], cfg['voc']['annotation_path_2012']]:
        for file_name in tqdm(os.listdir(annotation_path)):
            xml_path = os.path.join(annotation_path, file_name)
            img_path = os.path.join(annotation_path.replace('Annotations', 'JPEGImages'), file_name.split(".")[0] + ".jpg")
            single_data = {
                "file_path": img_path,
                "labels": []
            }
            root = ET.parse(xml_path).getroot()
            for obj in root.iter("object"):
                obj_name = obj.find("name").text.strip() # type: ignore
                bndbox = obj.find("bndbox")
                box_part = ["xmin", "ymin", "xmax", "ymax"]
                label = {}
                for part in box_part:
                    label[part] = bndbox.find(part).text # type: ignore
                label["name_id"] = name2id[obj_name]
                single_data["labels"].append(label)
            big_labels_data.append(single_data)

    # 打乱数据，拆分为训练集和验证集
    random.shuffle(big_labels_data)
    split_index = int(len(big_labels_data) * cfg['ds']['split_ratio'])
    train_data = big_labels_data[:split_index]
    val_data = big_labels_data[split_index:]

    with open(out_train, 'w', encoding='utf-8') as f:
        train_json = json.dumps(train_data, ensure_ascii=False, indent=2)
        logging.info("正在保存训练集数据")
        f.write(train_json)

    with open(out_val, 'w', encoding='utf-8') as f:
        val_json = json.dumps(val_data, ensure_ascii=False, indent=2)
        logging.info("正在保存验证集数据")
        f.write(val_json)


if __name__ == "__main__":
    out_train = 'voc_train.json'
    out_val = 'voc_val.json'
    with open('config/yolo_conf.yaml', encoding="ascii", errors="ignore")as f:
        cfg = yaml.safe_load(f)
    main(cfg, out_train, out_val)
