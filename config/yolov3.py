import torch

class MyConf():
        imgsize = 416  # 网络输入大小
        sample_ratio = [32, 16, 8]
        feature_map = [13, 26, 52]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchsize = 2
        learning_rate = 1e-3
        weight_decay = 1e-3
        epochs = 100

        def __init__(self):
            super(MyConf, self).__init__()
            # 获取anchors
            self.anchors = []
            with open('config/anchors.txt', 'r', encoding='utf-8') as f:
                temp = []
                for item in enumerate(f.readline().strip('\n').split(' ')):
                    index, anchor = item
                    temp.append([int(i) for i in anchor.split(',')])
                    if (index + 1) % 3 == 0:
                        self.anchors.append(temp)
                        temp = []
                        continue

            # 获取class类别名字
            self.class_name = []
            with open('config/coco_classes.txt', 'r', encoding='utf-8') as f:
                for i in f.readlines():
                    self.class_name.append(i.strip('\n'))
CONF = MyConf()