from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        filter_in,
                        filter_out,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=pad,
                        bias=False,
                    ),
                ),
                ("bn", nn.BatchNorm2d(filter_out)),
                ("relu", nn.LeakyReLU(0.1)),
            ]
        )
    )


# ------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
# ------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(
            filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True
        ),
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors, anchors_mask, class_name, pretrained=False):
        super().__init__()
        # 注册基本参数
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_name = class_name
        self.num_classes = len(class_name)
        # ---------------------------------------------------#
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        # ---------------------------------------------------#
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(
                torch.load("model_data/darknet53_backbone_weights.pth")
            )

        # ---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        # ---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        # ------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        # ------------------------------------------------------------------------#
        self.last_layer0 = make_last_layers(
            [512, 1024], out_filters[-1], len(anchors_mask[0]) * (self.num_classes + 5)
        )

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.last_layer1 = make_last_layers(
            [256, 512],
            out_filters[-2] + 256,
            len(anchors_mask[1]) * (self.num_classes + 5),
        )

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.last_layer2 = make_last_layers(
            [128, 256],
            out_filters[-3] + 128,
            len(anchors_mask[2]) * (self.num_classes + 5),
        )

    def forward(self, x):
        # ---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        # ---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        # ---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        # ---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)
        # 转换层
        # (batch_size, 255, 13, 13) -> (batch_size, 13, 13, 255) -> (batch_size, 13, 13, 3, num_classes + 5) -> (batch_size, 3, 13, 13, num_classes + 5)
        out0 = (
            out0.permute(0, 2, 3, 1)
            .reshape(
                -1,
                out0.size(2),
                out0.size(3),
                len(self.anchors_mask[0]),
                self.num_classes + 5,
            )
            .permute(0, 3, 1, 2, 4)
        )

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        # ---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        # ---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)
        # 转换层
        # (batch_size, 255, 26, 26) -> (batch_size, 26, 26, 255) -> (batch_size, 26, 26, 3, num_classes + 5) -> (batch_size, 3, 26, 26, num_classes + 5)
        out1 = (
            out1.permute(0, 2, 3, 1)
            .reshape(
                -1,
                out1.size(2),
                out1.size(3),
                len(self.anchors_mask[1]),
                self.num_classes + 5,
            )
            .permute(0, 3, 1, 2, 4)
        )
        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        # ---------------------------------------------------#
        #   第三个特征层
        #   out3 = (batch_size,255,52,52)
        # ---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        # 转换层
        # (batch_size, 255, 52, 52) -> (batch_size, 52, 52, 255) -> (batch_size, 52, 52, 3, num_classes + 5) -> (batch_size, 3, 52, 52, num_classes + 5)
        out2 = (
            out2.permute(0, 2, 3, 1)
            .reshape(
                -1,
                out2.size(2),
                out2.size(3),
                len(self.anchors_mask[2]),
                self.num_classes + 5,
            )
            .permute(0, 3, 1, 2, 4)
        )

        return out2, out1, out0
