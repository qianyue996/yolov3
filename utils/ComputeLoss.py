import torch
import torch.nn as nn


class YOLOv3LOSS:
    def __init__(self, device, l_loc, l_cls, l_obj, l_noo):
        self.device = device

        self.lambda_obj_layers = [4.0, 1.0, 0.4]

        self.lambda_loc = l_loc
        self.lambda_cls = l_cls
        self.lambda_obj = l_obj
        self.lambda_noo = l_noo

    def __call__(self, model, predict, targets):
        nl = len(predict)
        all_loss_loc = torch.zeros(1).to(self.device)
        all_loss_cls = all_loss_loc.clone()
        all_loss_obj = all_loss_loc.clone()
        all_loss_noo = all_loss_loc.clone()
        # ===========================================#
        #
        # ===========================================#
        for i in range(nl):
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
            stride = model.model[-1].stride[i]
            # ===========================================#
            #   加载anchors锚框
            # ===========================================#
            anchors = model.model[-1].anchors[i]
            # ===========================================#
            #   重构网络预测结果
            #   shape: (B, 3, S, S, 5+80) coco
            # ===========================================#
            prediction = predict[i]
            # ===========================================#
            #   构建网络应有的预测结果y_true
            #   shape: (B, 3, S, S, 5+80)
            # ===========================================#
            y_true = self.build_target(B, S, prediction, targets, anchors)
            # ===========================================#
            #   正样本mask
            # ===========================================#
            obj_mask = y_true[..., 4] == 1

            x = prediction[obj_mask][:, 0].sigmoid() * S

            y = prediction[obj_mask][:, 1].sigmoid() * S

            best_a = obj_mask.nonzero()[:, 1]

            w = torch.exp(prediction[obj_mask][:, 2]) * anchors[best_a][:, 0] / stride

            h = torch.exp(prediction[obj_mask][:, 3]) * anchors[best_a][:, 1] / stride
            # ===========================================#
            #   负样本mask
            # ===========================================#
            noobj_mask = self.ignore_target(obj_mask)

            if obj_mask.sum().item() != 0:
                _cls = torch.sigmoid(prediction[..., 5:])[obj_mask]
                t_cls = y_true[..., 5:][obj_mask]
                # ===========================================#
                #   位置损失 GIoU损失
                # ===========================================#
                giou = self.compute_giou(x, y, w, h, obj_mask, anchors, y_true, S)
                loss_loc = (1 - giou).mean()
                # self.compute_mseloss(x, y, w, h, y_true, obj_mask, S, anchors, stride)
                all_loss_loc += loss_loc
                # ===========================================#
                #   分类损失
                # ===========================================#
                loss_cls = nn.BCELoss(reduction="mean")(_cls, t_cls)
                all_loss_cls += loss_cls
            # ===========================================#
            #   置信度损失
            # ===========================================#
            conf = torch.sigmoid(prediction[..., 4])
            t_conf = y_true[..., 4]
            loss_conf = nn.BCELoss(reduction="none")(conf, t_conf)
            # ===========================================#
            #   GroundTrue Postivez正样本置信度损失
            # ===========================================#
            if obj_mask.sum() != 0:
                obj_conf = loss_conf[obj_mask].mean() * self.lambda_obj_layers[i]
                all_loss_obj += obj_conf
            # ===========================================#
            #   Background Negative负样本置信度损失
            # ===========================================#
            noobj_conf = loss_conf[noobj_mask].mean() * self.lambda_obj_layers[i]
            all_loss_noo += noobj_conf
        # ===========================================#
        #   没加lambda系数的loss，方便观察loss下降情况
        # ===========================================#
        original_loss_loc = all_loss_loc
        original_loss_cls = all_loss_cls
        original_loss_obj = all_loss_obj
        original_loss_noo = all_loss_noo
        # ===========================================#
        #   计算总loss
        # ===========================================#
        all_loss_loc *= self.lambda_loc
        all_loss_cls *= self.lambda_cls
        all_loss_obj *= self.lambda_obj
        all_loss_noo *= self.lambda_noo

        loss = all_loss_loc + all_loss_cls + all_loss_obj + all_loss_noo

        return {
            "loss": loss,
            "loss_loc": original_loss_loc,
            "loss_cls": original_loss_cls,
            "loss_obj": original_loss_obj,
            "loss_noo": original_loss_noo,
            "lambda_loc": self.lambda_loc,
            "lambda_cls": self.lambda_cls,
            "lambda_obj": self.lambda_obj,
        }

    def _fill_target(self, y_true, bs, k, x, y, batch_target, anchors, c, index):
        """辅助函数：填充目标值"""
        y_true[bs, k, x, y, 0] = batch_target[index, 0] - x.float()
        y_true[bs, k, x, y, 1] = batch_target[index, 1] - y.float()
        y_true[bs, k, x, y, 2] = torch.log(batch_target[index, 2] / anchors[k][0])
        y_true[bs, k, x, y, 3] = torch.log(batch_target[index, 3] / anchors[k][1])
        y_true[bs, k, x, y, 4] = 1
        y_true[bs, k, x, y, 5 + c] = 1
        return y_true

    def build_target(self, B, S, prediction, targets, anchors):
        y_true = torch.zeros_like(prediction).to(self.device)

        for bs in range(B):
            if targets[bs].shape[0] == 0:  # 处理无目标的情况
                continue

            # 预处理目标框
            batch_target = torch.zeros_like(targets[bs])
            batch_target[:, 0:4] = targets[bs][:, 0:4] * S
            batch_target[:, 4] = targets[bs][:, 4]

            # 计算IOU矩阵
            gt_box = batch_target[:, 2:4]
            iou_matrix = self.compute_iou(gt_box, anchors)
            best_iou, best_na = torch.max(iou_matrix, dim=1)

            # 处理每个目标框
            for index, n_a in enumerate(best_na.tolist()):
                # 获取坐标和类别
                x = torch.clamp(batch_target[index, 0].long(), 0, S - 1)
                y = torch.clamp(batch_target[index, 1].long(), 0, S - 1)
                c = batch_target[index, 4].long()

                # 处理最佳匹配的anchor
                k = n_a
                y_true = self._fill_target(y_true, bs, k, x, y, batch_target, anchors, c, index)

                # 处理其他匹配的anchor
                additional_anchors = (iou_matrix[index] > 0.5).nonzero().squeeze(1)
                for a in additional_anchors.tolist():
                    if a != n_a:
                        k = a
                        y_true = self._fill_target(y_true, bs, k, x, y, batch_target, anchors, c, index)

        return y_true

    def compute_iou(self, gt_box, anchors):
        gt_box = gt_box.unsqueeze(1)
        anchors = anchors.unsqueeze(0)

        min_wh = torch.min(gt_box, anchors)
        area = min_wh[..., 0] * min_wh[..., 1]

        area_a = (gt_box[..., 0] * gt_box[..., 1]).expand_as(area)
        area_b = (anchors[..., 0] * anchors[..., 1]).expand_as(area)
        union = area_a + area_b - area

        return area / union

    def compute_giou(self, x, y, w, h, obj_mask, anchors, y_true, S):
        b, a, gj, gi = obj_mask.nonzero(as_tuple=True)
        anchor = anchors[a]

        t_x = torch.sigmoid(y_true[obj_mask])[:, 0] * S
        t_y = torch.sigmoid(y_true[obj_mask])[:, 1] * S
        t_w = torch.exp(y_true[obj_mask][:, 2]) * anchor[:, 0]
        t_h = torch.exp(y_true[obj_mask][:, 3]) * anchor[:, 1]

        # xywh -> xyxy
        x1 = torch.clamp(x - w / 2, min=1e-6, max=S)
        y1 = torch.clamp(y - h / 2, min=1e-6, max=S)
        x2 = torch.clamp(x + w / 2, min=1e-6, max=S)
        y2 = torch.clamp(y + h / 2, min=1e-6, max=S)

        t_x1 = torch.clamp(t_x - t_w / 2, min=1e-6, max=S)
        t_y1 = torch.clamp(t_y - t_h / 2, min=1e-6, max=S)
        t_x2 = torch.clamp(t_x + t_w / 2, min=1e-6, max=S)
        t_y2 = torch.clamp(t_y + t_h / 2, min=1e-6, max=S)

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

        giou = iou - (area_c - union) / (area_c.clamp(min=1e-6) + 1e-16)

        return giou

    def ignore_target(self, obj_mask):
        # ===========================================#
        #   batch size
        # ===========================================#
        B = obj_mask.shape[0]
        # ===========================================#
        #   构建noobj mask
        # ===========================================#
        noobj_mask = torch.ones_like(obj_mask, dtype=torch.bool).to(self.device)
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

    def compute_mseloss(self, x, y, w, h, y_true, obj_mask, S, anchors, stride):
        best_a = obj_mask.nonzero()[:, 1]
        x_loss = ((x - y_true[obj_mask][:, 0] * S) ** 2).sum()
        y_loss = ((y - y_true[obj_mask][:, 1] * S) ** 2).sum()
        w_loss = ((w - torch.exp(y_true[obj_mask][:, 2]) * anchors[best_a][:, 1] / stride) ** 2).sum()
        h_loss = ((h - torch.exp(y_true[obj_mask][:, 3]) * anchors[best_a][:, 0] / stride) ** 2).sum()
        return x_loss + y_loss + w_loss + h_loss