import torch
import torch.nn as nn


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(CBL, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.layers(x)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.layer1 = CBL(3, 16, 3, 1, 1)
        self.layer2 = nn.MaxPool2d(2, 2)

        self.layer3 = CBL(16, 32, 3, 1, 1)
        self.layer4 = nn.MaxPool2d(2, 2)

        self.layer5 = CBL(32, 64, 3, 1, 1)
        self.layer6 = nn.MaxPool2d(2, 2)

        self.layer7 = CBL(64, 128, 3, 1, 1)
        self.layer8 = nn.MaxPool2d(2, 2)

        self.layer9 = CBL(128, 256, 3, 1, 1)
        self.layer10 = nn.MaxPool2d(2, 2)

        self.layer11 = CBL(256, 512, 3, 1, 1)
        self.layer12 = nn.ZeroPad2d([0, 1, 0, 1])
        self.layer13 = nn.MaxPool2d(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x_small = x
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x_large = self.layer13(x)
        return x_small, x_large


class YOLOv3Head(nn.Module):
    def __init__(self, num_classes=None):
        super(YOLOv3Head, self).__init__()
        self.layer1 = CBL(512, 1024, 3, 1, 1)
        self.layer2 = CBL(1024, 256, 1, 1, 0)
        self.layer3 = CBL(256, 512, 3, 1, 1)
        self.layer4 = CBL(256, 128, 1, 1, 0)
        self.layer5 = nn.Upsample(None, scale_factor=2, mode="nearest")
        self.layer6 = CBL(256, 256, 3, 1, 1)
        self.layer7 = nn.Conv2d(512, 3 * (5 + num_classes), 1, 1, 0)
        self.layer8 = nn.Conv2d(256, 3 * (5 + num_classes), 1, 1, 0)

    def forward(self, x_small, x_large):
        x = self.layer1(x_large)
        x = self.layer2(x)
        x_small_upsample = x
        x = self.layer3(x)
        y_large = x
        x = self.layer4(x_small_upsample)
        x = self.layer5(x)
        x = torch.cat([x, x_small], dim=1)
        y_small = self.layer6(x)

        y_large = self.layer7(y_large)
        y_small = self.layer8(y_small)

        return y_large, y_small


class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=None):
        super(YOLOv3Tiny, self).__init__()
        self.backbone = Backbone()
        self.head = YOLOv3Head(num_classes=num_classes)

    def forward(self, x):
        x_small, x_large = self.backbone(x)
        x_large, x_small = self.head(x_small, x_large)

        return x_large, x_small


if __name__ == "__main__":
    x = torch.randn(2, 3, 416, 416)
    model = YOLOv3Tiny(num_classes=20)
    y = model(x)
    print(model.parameters())
    print(model.parameters())
