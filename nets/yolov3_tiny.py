import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # 第一特征提取路径 (输出26x26)
        self.layer1 = nn.Sequential(
            ConvBlock(3, 16, 3),
            nn.MaxPool2d(2, 2),  # 208x208
            ConvBlock(16, 32, 3),
            nn.MaxPool2d(2, 2),  # 104x104
            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2, 2),  # 52x52
            ConvBlock(64, 128, 3),
            nn.MaxPool2d(2, 2),  # 26x26
            ConvBlock(128, 256, 3),  # 保持26x26
        )

        # 第二特征提取路径 (输出13x13)
        self.layer2 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2),  # 下采样到13x13
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 256, 1),
        )

    def forward(self, x):
        feat_small = self.layer1(x)  # [batch, 256, 26, 26]
        feat_large = self.layer2(feat_small)  # [batch, 256, 13, 13]
        return feat_small, feat_large


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=None):
        super(YOLOv3Tiny, self).__init__()
        self.backbone = Backbone()

        # 第一个检测层（处理13x13特征）
        self.detection1 = nn.Sequential(
            ConvBlock(256, 512, 3), nn.Conv2d(512, 3 * (5 + num_classes), 1)
        )

        # 第二个检测层（处理26x26特征）
        self.detection2_conv = ConvBlock(512, 256, 3)  # 通道数修正为512→256
        self.detection2_final = nn.Conv2d(256, 3 * (5 + num_classes), 1)

    def forward(self, x):
        feat_small, feat_large = self.backbone(x)

        # 第一个检测分支
        out1 = self.detection1(feat_large)

        # 第二个检测分支
        x2_up = nn.Upsample(scale_factor=2, mode="nearest")(feat_large)  # 13→26
        combined = torch.cat([feat_small, x2_up], dim=1)  # 256+256=512通道
        out2 = self.detection2_conv(combined)
        out2 = self.detection2_final(out2)

        return out1, out2


if __name__ == "__main__":
    x = torch.randn(1, 3, 416, 416)
    model = YOLOv3Tiny(num_classes=20)
    y = model(x)
    print(model.parameters())
