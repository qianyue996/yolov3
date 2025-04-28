import json
import os
import xml.etree.ElementTree as ET

import tqdm

with open('config/datasetParameter.json', 'r', encoding='utf-8')as f:
    config = json.load(f)

image_path = r'D:\Python\yolov1\data\VOCdevkit\VOC2012\JPEGImages'
annotation_path = r'D:\Python\yolov1\data\VOCdevkit\VOC2012\Annotations'

with open('voc_train.txt', 'w', encoding='utf-8')as f:
    for i, name in enumerate(tqdm.tqdm(os.listdir(annotation_path))):
        xml_path = os.path.join(annotation_path, name)
        img_path = os.path.join(image_path, name.split('.')[0]+'.jpg')
        f.write(img_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for _object in root.findall('object'):
            name = _object.find('name').text
            _id = config['voc']['class_name'].index(name)
            bndbox = _object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            bndbox = list(map(float, [xmin, ymin, xmax, ymax]))
            f.write(f' {xmin},{ymin},{xmax},{ymax},{_id}')
        f.write('\n')