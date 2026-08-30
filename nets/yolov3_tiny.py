from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm


class CBL(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1
    ) -> None:
        super().__init__()
        padding = (kernel - 1) // 2
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = CBL(3, 16, 3, 1)
        self.layer2 = nn.MaxPool2d(2, 2)

        self.layer3 = CBL(16, 32, 3, 1)
        self.layer4 = nn.MaxPool2d(2, 2)

        self.layer5 = CBL(32, 64, 3, 1)
        self.layer6 = nn.MaxPool2d(2, 2)

        self.layer7 = CBL(64, 128, 3, 1)
        self.layer8 = nn.MaxPool2d(2, 2)

        self.layer9 = CBL(128, 256, 3, 1)
        self.layer10 = nn.MaxPool2d(2, 2)

        self.layer11 = CBL(256, 512, 3, 1)
        self.layer12 = nn.ZeroPad2d((0, 1, 0, 1))
        self.layer13 = nn.MaxPool2d(2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class YOLOv3Neck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = CBL(512, 1024, 3, 1)
        self.layer2 = CBL(1024, 256, 1, 1)
        self.layer3 = CBL(256, 512, 3, 1)
        self.layer4 = CBL(256, 128, 1, 1)
        self.layer5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.layer6 = CBL(256, 256, 3, 1)

    def forward(
        self, x_small: torch.Tensor, x_large: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x_large)
        x = self.layer2(x)
        x_small_upsample = x
        x = self.layer3(x)
        x_large = x
        x = self.layer4(x_small_upsample)
        x = self.layer5(x)
        x = torch.cat([x, x_small], dim=1)
        x_small = self.layer6(x)

        return x_large, x_small


class YOLOv3Head(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_anchors_large: int = 3,
        num_anchors_small: int = 3,
    ) -> None:
        super().__init__()
        self.layer_large = nn.Conv2d(512, num_anchors_large * (5 + num_classes), 1, 1)
        self.layer_small = nn.Conv2d(256, num_anchors_small * (5 + num_classes), 1, 1)

    def forward(
        self, x_large: torch.Tensor, x_small: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_large = self.layer_large(x_large)
        x_small = self.layer_small(x_small)

        return x_large, x_small


class YOLOv3Tiny(nn.Module):
    """YOLOv3-Tiny 轻量版检测模型（2 个检测尺度，stride 16 与 stride 32）。"""

    def __init__(
        self,
        anchors: list[list[int]],
        anchors_mask: list[list[int]],
        class_names: list[str],
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        # 注册与 YoloBody 完全一致的基本参数
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.backbone = Backbone()
        self.neck = YOLOv3Neck()
        self.head = YOLOv3Head(
            num_classes=self.num_classes,
            num_anchors_large=len(anchors_mask[0]),
            num_anchors_small=len(anchors_mask[1]) if len(anchors_mask) > 1 else len(anchors_mask[0]),
        )

        # Focal Loss 专属初始先验偏置 (RetinaNet 论文第 4.1 节)：
        # 将检测头置信度输出通道的 bias 初始化为 b = -4.6 (即 pi = 0.01)
        for layer in [self.head.layer_large, self.head.layer_small]:
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                b = layer.bias.view(-1, self.num_classes + 5)
                b.data[:, 4] = -4.6

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_small, x_large = self.backbone(x)
        x_large, x_small = self.neck(x_small, x_large)
        out_large, out_small = self.head(x_large, x_small)

        # 转换为标准 (B, nA, H, W, 5+C) 输出格式
        out_large = (
            out_large.permute(0, 2, 3, 1)
            .reshape(
                -1,
                out_large.size(2),
                out_large.size(3),
                len(self.anchors_mask[0]),
                self.num_classes + 5,
            )
            .permute(0, 3, 1, 2, 4)
        )
        out_small = (
            out_small.permute(0, 2, 3, 1)
            .reshape(
                -1,
                out_small.size(2),
                out_small.size(3),
                len(self.anchors_mask[1]) if len(self.anchors_mask) > 1 else len(self.anchors_mask[0]),
                self.num_classes + 5,
            )
            .permute(0, 3, 1, 2, 4)
        )

        # 从小尺度 stride 到大尺度 stride: 26x26 (stride 16), 13x13 (stride 32)
        return out_small, out_large


if __name__ == "__main__":
    anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]
    anchors_mask = [[3, 4, 5], [0, 1, 2]]
    class_names = ["person", "dog"]

    x = torch.randn(2, 3, 416, 416)
    model = YOLOv3Tiny(
        anchors=anchors, anchors_mask=anchors_mask, class_names=class_names
    )
    small, large = model(x)
    tqdm.write(f"Tiny output shapes: small={small.shape}, large={large.shape}")
