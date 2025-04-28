import json

import torch
import torch.nn as nn


class YOLOv3LOSS:
    def __init__(self, device, l_loc, l_cls, l_obj):
        self.device = device
        self.stride = [32, 16, 8]
        self.anchors = [
            [7, 9],
            [16, 24],
            [43, 26],
            [29, 60],
            [72, 56],
            [63, 133],
            [142, 96],
            [166, 223],
            [400, 342],
        ]
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.lambda_obj_layers = [0.4, 1.0, 4]

        self.lambda_loc = l_loc
        self.lambda_cls = l_cls
        self.lambda_obj = l_obj

    def __call__(self, predict, targets):
        all_loss_loc = torch.zeros(1, device=self.device)
        all_loss_cls = all_loss_loc.clone()
        all_loss_obj = all_loss_loc.clone()
        # ===========================================#
        #   循环3个Layers，对应yolov3的三个特征层
        # ===========================================#
        for i in range(3):
            # ===========================================#
            #   参数
            # ===========================================#
            B = predict[i].shape[0]
            # ===========================================#
            #   获取feature map的宽高
            #   将图像分割为SxS个网格
            #   13,26,52
            # ===========================================#
            S = predict[i].shape[2]
            # ===========================================#
            #   获取每个网格的步长
            #   13,26,52 ==> 32,16,8
            # ===========================================#
            stride = self.stride[i]
            # ===========================================#
            #   加载anchors锚框
            # ===========================================#
            anchors = torch.tensor(
                self.anchors, dtype=torch.float32, device=self.device
            )
            # ===========================================#
            #   重构网络预测结果
            #   shape: (B, 3, S, S, 5+80) coco
            # ===========================================#
            prediction = predict[i].view(B, 3, 5 + 80, S, S).permute(0, 1, 3, 4, 2)
            # ===========================================#
            #   构建网络应有的预测结果y_true
            #   shape: (B, 3, S, S, 5+80)
            # ===========================================#
            y_true = self.build_target(i, B, S, targets, anchors)

            x = torch.sigmoid(prediction[..., 0])

            y = torch.sigmoid(prediction[..., 1])

            w = prediction[..., 2]

            h = prediction[..., 3]

            conf = torch.sigmoid(prediction[..., 4])
            t_conf = y_true[..., 4]
            # ===========================================#
            #   正样本mask
            # ===========================================#
            obj_mask = y_true[..., 4] == 1
            # ===========================================#
            #   负样本mask
            # ===========================================#
            noobj_mask = self.ignore_target(obj_mask)

            if obj_mask.sum() != 0:

                _cls = torch.sigmoid(prediction[..., 5:])[obj_mask]
                t_cls = y_true[..., 5:][obj_mask]

                giou = self.compute_giou(x, y, w, h, obj_mask, anchors, stride, y_true)
                # ===========================================#
                #   位置损失 GIoU损失
                # ===========================================#
                loss_loc = (1 - giou).mean()
                all_loss_loc += loss_loc
                # ===========================================#
                #   分类损失
                # ===========================================#
                loss_cls = nn.BCELoss(reduction="mean")(_cls, t_cls)
                all_loss_cls += loss_cls
            # ===========================================#
            #   置信度损失
            # ===========================================#
            loss_conf = nn.BCELoss(reduction="none")(conf, t_conf)
            loss_conf = (
                loss_conf[obj_mask | noobj_mask].mean() * self.lambda_obj_layers[i]
            )
            all_loss_obj += loss_conf
            # ===========================================#
            #   GroundTrue Postivez正样本置信度损失
            # ===========================================#
            # obj_conf = loss_conf[obj_mask].mean() * self.conf_lambda[i]
            # all_obj_conf += torch.nan_to_num(obj_conf, nan=0.0)
            # ===========================================#
            #   Background Negative负样本置信度损失
            # ===========================================#
            # noobj_conf = loss_conf[noobj_mask].mean() * self.conf_lambda[i]
            # all_noobj_conf += noobj_conf
        # ===========================================#
        #   计算总loss
        # ===========================================#
        all_loss_loc *= self.lambda_loc
        all_loss_cls *= self.lambda_cls
        all_loss_obj *= self.lambda_obj
        loss = all_loss_loc + all_loss_obj + all_loss_cls

        return {
            "loss": loss,
            "loss_loc": all_loss_loc,
            "loss_obj": all_loss_obj,
            "loss_cls": all_loss_cls,
            "positive_num": obj_mask.sum(),
            "lambda_loc": self.lambda_loc,
            "lambda_cls": self.lambda_cls,
            "lambda_obj": self.lambda_obj,
        }

    def build_target(self, i, B, S, targets, anchors):
        y_true = torch.zeros(B, 3, S, S, 5 + 80, device=self.device)

        for bs in range(B):
            if targets[bs].shape[0] == 0:  # 处理无目标的情况
                continue

            batch_target = torch.zeros_like(targets[bs])
            batch_target[:, 0:2] = targets[bs][:, 0:2] * S
            batch_target[:, 2:4] = targets[bs][:, 2:4] * 416
            batch_target[:, 4] = targets[bs][:, 4]

            gt_box = batch_target[:, 2:4]
            best_iou, best_na = torch.max(self.compute_iou(gt_box, anchors), dim=1)

            for index, n_a in enumerate(best_na.tolist()):
                if n_a not in self.anchors_mask[i]:
                    continue

                k = self.anchors_mask[i].index(n_a)
                x = torch.clamp(batch_target[index, 0].long(), 0, S - 1)
                y = torch.clamp(batch_target[index, 1].long(), 0, S - 1)
                c = batch_target[index, 4].long()

                y_true[bs, k, x, y, 0] = batch_target[index, 0] - x.float()
                y_true[bs, k, x, y, 1] = batch_target[index, 1] - y.float()
                y_true[bs, k, x, y, 2] = torch.log(
                    batch_target[index, 2] / anchors[n_a][0]
                )
                y_true[bs, k, x, y, 3] = torch.log(
                    batch_target[index, 3] / anchors[n_a][1]
                )
                y_true[bs, k, x, y, 4] = 1
                y_true[bs, k, x, y, 5 + c] = 1

        return y_true

    def compute_iou(self, gt_box, anchors):
        gt_box = gt_box.unsqueeze(1)
        anchors = anchors.unsqueeze(0)

        min_wh = torch.min(gt_box, anchors)
        iou_i = min_wh[..., 0] * min_wh[..., 1]

        area_a = (gt_box[..., 0] * gt_box[..., 1]).expand_as(iou_i)
        area_b = (anchors[..., 0] * anchors[..., 1]).expand_as(iou_i)
        iou_u = area_a + area_b - iou_i

        return iou_i / iou_u

    def compute_giou(self, x, y, w, h, obj_mask, anchors, stride, y_true):
        best_a = obj_mask.nonzero()[:, 1]
        grid_x = obj_mask.nonzero()[:, 2]
        grid_y = obj_mask.nonzero()[:, 3]

        x = (x[obj_mask] + grid_x) * stride
        y = (y[obj_mask] + grid_y) * stride
        w = torch.exp(w[obj_mask]) * anchors[best_a][:, 0]
        h = torch.exp(h[obj_mask]) * anchors[best_a][:, 1]

        t_x = (y_true[obj_mask][:, 0] + grid_x) * stride
        t_y = (y_true[obj_mask][:, 1] + grid_y) * stride
        t_w = torch.exp(y_true[obj_mask][:, 2]) * anchors[best_a][:, 0]
        t_h = torch.exp(y_true[obj_mask][:, 3]) * anchors[best_a][:, 1]

        # xywh -> xyxy
        x1 = torch.clamp(x - w / 2, min=1e-6, max=416)
        y1 = torch.clamp(y - h / 2, min=1e-6, max=416)
        x2 = torch.clamp(x + w / 2, min=1e-6, max=416)
        y2 = torch.clamp(y + h / 2, min=1e-6, max=416)

        t_x1 = torch.clamp(t_x - t_w / 2, min=1e-6, max=416)
        t_y1 = torch.clamp(t_y - t_h / 2, min=1e-6, max=416)
        t_x2 = torch.clamp(t_x + t_w / 2, min=1e-6, max=416)
        t_y2 = torch.clamp(t_y + t_h / 2, min=1e-6, max=416)

        # 计算交集区域面积
        area_x1 = torch.max(x1, t_x1)
        area_y1 = torch.max(y1, t_y1)
        area_x2 = torch.min(x2, t_x2)
        area_y2 = torch.min(y2, t_y2)

        inter = (area_x2 - area_x1).clamp(min=0) * (area_y2 - area_y1).clamp(min=0)
        area_1 = (x2 - x1) * (y2 - y1)
        area_2 = (t_x2 - t_x1) * (t_y2 - t_y1)
        union = area_1 + area_2 - inter

        iou = inter / union.clamp(min=1e-6)

        # 最小包围框
        xc1 = torch.min(x1, t_x1)
        yc1 = torch.min(y1, t_y1)
        xc2 = torch.max(x2, t_x2)
        yc2 = torch.max(y2, t_y2)
        area_c = (xc2 - xc1) * (yc2 - yc1)

        giou = iou - (area_c - union) / area_c.clamp(min=1e-6)

        return giou

    def ignore_target(self, obj_mask):
        # ===========================================#
        #   batch size
        # ===========================================#
        B = obj_mask.shape[0]
        # ===========================================#
        #   构建noobj mask
        # ===========================================#
        noobj_mask = torch.ones_like(obj_mask, dtype=torch.bool, device=obj_mask.device)
        # ===========================================#
        #   遍历每一个正样本
        #   在正样本位置上的anchors 都置为0
        # ===========================================#
        for bs in range(B):
            if obj_mask[bs].sum().item() == 0:
                continue
            # ===========================================#
            #   正样本数量
            # ===========================================#
            N = obj_mask[bs].sum().item()
            for n in range(N):
                # ===========================================#
                #   获取正样本的x坐标
                # ===========================================#
                x = obj_mask[bs].nonzero()[n][-2:][0].item()
                # ===========================================#
                #   获取正样本的y坐标
                # ===========================================#
                y = obj_mask[bs].nonzero()[n][-2:][1].item()

                noobj_mask[bs, :, x, y] = False

        return noobj_mask
