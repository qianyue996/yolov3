#-------------------------------------------------------#
#   用于处理COCO数据集，根据json文件生成txt文件用于训练
#-------------------------------------------------------#
import json
import os
from collections import defaultdict
from tqdm import tqdm

class PreData():
    def __init__(self):
        super(PreData, self).__init__()
        #-------------------------------------------------------#
        #   指向了COCO训练集与验证集图片的路径
        #-------------------------------------------------------#
        self.train_datasets_path     = r"D:\Python\datasets\coco2014\train2014"
        self.val_datasets_path       = r"D:\Python\datasets\coco2014\val2014"

        #-------------------------------------------------------#
        #   指向了COCO训练集与验证集标签的路径
        #-------------------------------------------------------#
        self.train_annotation_path   = r"D:\Python\datasets\coco2014\annotations\instances_train2014.json"
        self.val_annotation_path     = r"D:\Python\datasets\coco2014\annotations\instances_val2014.json"

        #-------------------------------------------------------#
        #   生成的txt文件路径
        #-------------------------------------------------------#
        self.train_output_path       = "coco_train.txt"
        self.val_output_path         = "coco_val.txt"

        #-------------------------------------------------------#
        #   类别txt文件路径
        #-------------------------------------------------------#
        self.coco_classes_path = r'config/coco_classes.txt'

        #-------------------------------------------------------#
        #   预加载数据
        #-------------------------------------------------------#
        
        with open(self.val_annotation_path, 'r', encoding='utf-8')as f:
            self.val_data = json.load(f)

        
        
    def __call__(self):
        #-------------------------------------------------------#
        #   train 定义变量
        #-------------------------------------------------------#
        name_box_id = defaultdict(list)

        #-------------------------------------------------------#
        #   正式处理数据
        #-------------------------------------------------------#
        with open(self.train_annotation_path, 'r', encoding='utf-8')as f:
            self.train_data = json.load(f)
        self.precoco_classes() # 处理coco2014数据集类别列表
        coco_classes = []
        with open(self.coco_classes_path, 'r', encoding='utf-8')as f:
            for name in f.readlines():
                coco_classes.append(name.strip('\n'))
        self.id2name = {item['id']:item['name'] for item in self.train_data['categories']}
        # 正式处理数据
        for data in tqdm(self.train_data['annotations'], desc='processing... '):
            image_id = data['image_id']
            full_path = os.path.join(self.train_datasets_path,
                                     f'COCO_train2014_{image_id:012d}.jpg')
            real_id = coco_classes.index(self.id2name[data['category_id']])
            name_box_id[full_path].append([data['bbox'], real_id])

        with open(self.train_output_path, 'w', encoding='utf-8')as f:
            for key in tqdm(name_box_id.keys(), desc='writing... '):
                f.write(key)
                box_infos = name_box_id[key]
                for info in box_infos:
                    x_min = int(info[0][0])
                    y_min = int(info[0][1])
                    x_max = x_min + int(info[0][2])
                    y_max = y_min + int(info[0][3])
                    box_info = f' {x_min},{y_min},{x_max},{y_max},{int(info[1])}'
                    f.write(box_info)
                f.write('\n')
        #-------------------------------------------------------#
        #   val 定义变量
        #-------------------------------------------------------#
        name_box_id = defaultdict(list)

    def precoco_classes(self):
        #-------------------------------------------------------#
        #   处理coco2014数据集
        #-------------------------------------------------------#
        coco_classes = sorted(name['name'] for name in self.train_data['categories'])
        with open(self.coco_classes_path, 'w', encoding='utf-8')as f:
            for i in coco_classes:
                f.write(f'{i}\n')
if __name__=='__main__':
    predata = PreData()
    predata()