import torch
import torch.nn as nn

from nets.darknet import darknet53


# ----------------------#
#   核心卷积模块（保持原样，正确无误）
# ----------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, include_activation=True):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1) if include_activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# ----------------------#
#   修正后的Neck模块（关键修正点）
# ----------------------#
class YOLONeck(nn.Module):
    def __init__(self, backbone_channels):
        super().__init__()

        # 深层特征处理路径（修正卷积顺序）
        self.process_1024 = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),  # 官方标准结构：1x1->3x3->1x1->3x3->1x1
        )

        # 上采样与特征融合（保持原样，正确）
        self.upsample_512 = nn.Sequential(
            ConvBlock(512, 256, 1, include_activation=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # 中层次特征处理（修正卷积顺序）
        self.process_768 = nn.Sequential(
            ConvBlock(768, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),  # 官方标准结构
        )

        self.upsample_256 = nn.Sequential(
            ConvBlock(256, 128, 1, include_activation=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # 浅层次特征处理（修正卷积顺序）
        self.process_384 = nn.Sequential(
            ConvBlock(384, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),  # 官方标准结构
        )

    def forward(self, features):
        x2, x1, x0 = features  # 输入顺序为浅->深 [256, 512, 1024]

        # 处理深层特征（1024通道）
        p5 = self.process_1024(x0)

        # 第一次上采样并与512通道特征融合
        up5 = self.upsample_512(p5)
        c4 = torch.cat([up5, x1], dim=1)  # 256+512=768
        p4 = self.process_768(c4)

        # 第二次上采样并与256通道特征融合
        up4 = self.upsample_256(p4)
        c3 = torch.cat([up4, x2], dim=1)  # 128+256=384
        p3 = self.process_384(c3)

        return p3, p4, p5  # 输出顺序：浅->深 [128, 256, 512]


# ----------------------#
#   Head模块（保持原样，正确无误）
# ----------------------#
class YOLOHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.predictors = nn.ModuleList(
            [
                nn.Conv2d(128, 3 * (5 + num_classes), 1),  # 对应浅层特征（大尺寸）
                nn.Conv2d(256, 3 * (5 + num_classes), 1),  # 中层特征
                nn.Conv2d(512, 3 * (5 + num_classes), 1),  # 深层特征（小尺寸）
            ]
        )

    def forward(self, p3, p4, p5):
        return [self.predictors[2](p5), self.predictors[1](p4), self.predictors[0](p3)]


# ----------------------#
#   完整YOLOv3模型（保持原样，正确无误）
# ----------------------#
class YOLOv3(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_weights.pth"))
        self.neck = YOLONeck(self.backbone.layers_out_filters)
        self.head = YOLOHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        neck_features = self.neck(features)
        return self.head(*neck_features)
