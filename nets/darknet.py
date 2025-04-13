import math
import torch.nn as nn

# 卷积层
class CBL(nn.Module):
    def __init__(self, inchannel, outchannel, k=3, stride=1, padding="same", bias=False) -> None:
        super(CBL, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, k, stride, padding, bias=bias),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.m(x)

# 残差层
class Residual(nn.Module):
    def __init__(self, inchannel) -> None:
        super(Residual, self).__init__()
        outchannel = inchannel // 2
        self.m = nn.Sequential(
            CBL(inchannel, outchannel, 1, 1, 0),
            CBL(outchannel, inchannel, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.m(x)

class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        # 定义darknet53的层数
        self.layoutNumber = [1, 2, 8, 8, 4]
        self.layerA = nn.Sequential(
            CBL(3, 32, 3, 1, 1),
            self.MultiResidual(32, 64, 1),
            self.MultiResidual(64, 128, 2),
            self.MultiResidual(128, 256, 8)
        )
        self.layerB = self.MultiResidual(256, 512, 8)
        self.layerC = self.MultiResidual(512, 1024, 4)

    def forward(self, x):
        out1 = self.layerA(x)
        out2 = self.layerB(out1)
        out3 = self.layerC(out2)
        return out1, out2, out3
    
    # 多层残差块
    def MultiResidual(self, inchannel, outchannel, count):
        t = []
        for i in range(count + 1):
            if i == 0:
                temp = CBL(inchannel, outchannel, 3, 2, 1)
            else:
                temp = Residual(outchannel)
            t.append(temp)
        return nn.Sequential(*t)