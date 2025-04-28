import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = self._make_conv(3, 16, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = self._make_conv(16, 32, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = self._make_conv(32, 64, 3, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = self._make_conv(64, 128, 3, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = self._make_conv(128, 256, 3, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = self._make_conv(256, 512, 3, 1)
        self.conv7 = self._make_conv(512, 1024, 3, 1)  # 移除了pool6层

    def _make_conv(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        feat1 = self.conv5(x)  # 浅层特征 [26x26x256]
        x = self.pool5(feat1)  # 13x13x256
        x = self.conv6(x)  # 13x13x512
        feat2 = self.conv7(x)  # 深层特征 [13x13x1024]
        return feat1, feat2


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.conv8 = self._make_conv(1024, 256, 1, 1)
        self.conv11 = self._make_conv(256, 128, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_conv(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, feat1, feat2):
        x = self.conv8(feat2)  # [13x13x1024 → 13x13x256]
        x = self.conv11(x)  # [13x13x256 → 13x13x128]
        x = self.upsample(x)  # [13x13x128 → 26x26x128]
        x = torch.cat([x, feat1], dim=1)  # 拼接后 [26x26x(128+256)=384]
        return x


class Head(nn.Module):
    def __init__(self, num_classes=80):
        super(Head, self).__init__()
        self.num_classes = num_classes
        # 修正输入通道为1024以匹配feat2的维度
        self.conv9 = self._make_conv(1024, 512, 3, 1)
        self.conv10 = nn.Conv2d(512, 3 * (5 + num_classes), 1, 1)
        self.conv12 = self._make_conv(384, 256, 3, 1)
        self.conv13 = nn.Conv2d(256, 3 * (5 + num_classes), 1, 1)

    def _make_conv(self, in_channels, out_channels, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, feat2, fused_feat):
        yolo1 = self.conv9(feat2)  # [13x13x1024 → 13x13x512]
        yolo1 = self.conv10(yolo1)  # [13x13x512 → 13x13x255]
        yolo2 = self.conv12(fused_feat)  # [26x26x384 → 26x26x256]
        yolo2 = self.conv13(yolo2)  # [26x26x256 → 26x26x255]
        return yolo1, yolo2


class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3Tiny, self).__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes)

    def forward(self, x):
        feat1, feat2 = self.backbone(x)
        fused_feat = self.neck(feat1, feat2)
        yolo1, yolo2 = self.head(feat2, fused_feat)
        return yolo1, yolo2


if __name__ == "__main__":
    model = YOLOv3Tiny()
    input_tensor = torch.randn(1, 3, 416, 416)
    yolo1, yolo2 = model(input_tensor)
    print(yolo1.shape)  # [1, 3*(5+80), 13, 13]
    print(yolo2.shape)  # [1, 3*(5+80), 26, 26]
