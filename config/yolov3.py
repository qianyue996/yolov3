import torch

class MyConf():
        imgsize = 416  # 网络输入大小
        sample_ratio = [32, 16, 8]
        feature_map = [13, 26, 52]
        anchors_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchsize = 2
        learning_rate = 1e-3
        weight_decay = 1e-3
        epochs = 100
        coco_train_path = 'coco_train.txt'
        coco_eval_path = 'coco_val.txt'

        def __init__(self):
            super(MyConf, self).__init__()
            # 获取anchors
            self.anchors = []
            with open('config/anchors.txt', 'r', encoding='utf-8') as f:
                for anchor in f.readline().strip('\n').split(' '):
                    w, h = anchor.split(',')
                    self.anchors.append([int(w), int(h)])

            # 获取class类别名字
            self.class_name = []
            with open('config/coco_classes.txt', 'r', encoding='utf-8') as f:
                for i in f.readlines():
                    self.class_name.append(i.strip('\n'))
CONF = MyConf()