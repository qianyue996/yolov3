import torch

class MyConf():
        imgsize = 416  # 网络输入大小
        anchorNumber = 3 # anchor框的多少
        net_scaled = [32, 16, 8]
        feature_map = [13, 26, 52]
        cocoPath = r"D:\Python\datasets\coco2014"
        vocPath = "F:/c/deepLearn/learn/pytorch/myYOLO/data/VOCdevkit/VOC2007"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchsize = 2
        weight_decay = 1e-3
        num_workers = 0
        epochs = 300
        anchorThes = 4
        valIOUTher = 0.5
        anchors = [
            [[116,90],  [156,198],  [373,326]], # 13*13
            [[30,61],  [62,45],  [59,119]], # 26*26
            [[10,13],  [16,30],  [33,23]], # 52*52
        ]
        anchorIndex = [[0,1,2],[3,4,5],[6,7,8]]

        cocoClass = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
        vocClass = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        classes = cocoClass
        classNumber = len(classes) # 种类数量 voc 20 coco 80
CONF = MyConf()