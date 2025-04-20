import torch

class MyConf():
        imgsize = 416  # 网络输入大小
        net_scaled = [32, 16, 8]
        feature_map = [13, 26, 52]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batchsize = 16
        learning_rate = 1e-3
        weight_decay = 1e-3
        num_workers = 0
        epochs = 100
        anchorThes = 4
        valIOUTher = 0.5

        per_feat_anc_num = 3 # 每个feature map anchor框的多少
        anchors = [
            [[142, 96], [166, 223],[400, 342]], # 13*13
            [[29, 60],  [72, 56],  [63, 133]], # 26*26
            [[7, 9],    [16, 24],  [43, 26]], # 52*52
        ]
        anchorIndex = [[0,1,2],[3,4,5],[6,7,8]]
CONF = MyConf()