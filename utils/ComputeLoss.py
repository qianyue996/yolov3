import torch
import torch.nn as nn
import numpy as np


class YOLOv3LOSS:
    def __init__(self, model, device, l_loc, l_cls, l_obj, l_noo):
        self.device = device
        self.anchors = model.model[-1].anchors
        self.na = self.anchors[0].shape[0]
        self.am = list(map(tuple, np.split(np.arange(self.anchors.view(-1, 2).shape[0]), self.anchors.view(-1, 2).shape[0] // self.anchors[0].shape[0])))

        self.lambda_obj_layers = [2.0, 0.4, 4.0]

        self.lambda_loc = l_loc
        self.lambda_cls = l_cls
        self.lambda_obj = l_obj
        self.lambda_noo = l_noo

    def __call__(self, predictions, targets):
        all_loss_loc = torch.zeros(1).to(self.device)
        all_loss_cls = all_loss_loc.clone()
        all_loss_obj = all_loss_loc.clone()
        all_loss_noo = all_loss_loc.clone()
        #============================================#
        #   拿基础参数
        #============================================#
        self._S = [i.shape[2] for i in predictions]
        for i, prediction in enumerate(predictions):
            #============================================#
            #   构建网络应有的预测结果y_true
            #   shape: (B, 3, S, S, 5+80)
            #============================================#
            y_true = self.build_target(i, prediction, targets)
            #============================================#
            #   正样本mask
            #============================================#
            obj_mask = y_true[..., 4] == 1
            #============================================#
            #   负样本mask
            #============================================#
            noobj_mask, pred_box, targ_box = self.ignore_target(i, prediction, y_true, obj_mask)

            if obj_mask.sum().item() != 0:
                #============================================#
                #   位置损失 GIoU损失
                #============================================#
                giou = self.compute_iou(pred_box, targ_box, iou_type='ciou')
                loss_loc = (1 - giou).mean()
                # self.compute_mseloss(x, y, w, h, y_true, obj_mask, S, anchors, stride)
                all_loss_loc += loss_loc
                #============================================#
                #   分类损失
                #============================================#
                _cls = prediction[..., 5:][obj_mask]
                t_cls = y_true[..., 5:][obj_mask]
                loss_cls = nn.BCEWithLogitsLoss(reduction='mean')(_cls, t_cls)
                all_loss_cls += loss_cls
            #============================================#
            #   置信度损失
            #============================================#
            # conf = prediction[..., 4][obj_mask | noobj_mask]
            # t_conf = y_true[..., 4][obj_mask | noobj_mask]
            # loss_conf = focal_loss(conf, t_conf) * self.lambda_obj_layers[i]
            # all_loss_obj += loss_conf
            conf = prediction[..., 4]
            t_conf = y_true[..., 4]
            loss_conf = nn.BCEWithLogitsLoss(reduction="none")(conf, t_conf)
            #============================================#
            #   置信度放一起计算
            #============================================#
            # all_loss_obj += loss_conf[obj_mask | noobj_mask].mean()
            #============================================#
            #   GroundTrue Postivez正样本置信度损失
            #============================================#
            if obj_mask.sum() != 0:
                obj_conf = loss_conf[obj_mask].mean() * self.lambda_obj_layers[i]
                all_loss_obj += obj_conf
            #============================================#
            #   Background Negative负样本置信度损失
            #============================================#
            noobj_conf = loss_conf[noobj_mask].mean() * self.lambda_obj_layers[i]
            all_loss_noo += noobj_conf
        #============================================#
        #   没加lambda系数的loss，方便观察loss下降情况
        #============================================#
        original_loss_loc = all_loss_loc.clone()
        original_loss_cls = all_loss_cls.clone()
        original_loss_obj = all_loss_obj.clone()
        original_loss_noo = all_loss_noo.clone()
        original_loss = original_loss_loc + original_loss_cls + original_loss_obj + original_loss_noo
        #============================================#
        #   计算总loss
        #============================================#
        all_loss_loc *= self.lambda_loc
        all_loss_cls *= self.lambda_cls
        all_loss_obj *= self.lambda_obj
        all_loss_noo *= self.lambda_noo

        loss = all_loss_loc + all_loss_cls + all_loss_obj + all_loss_noo

        return {
            "loss": loss,
            "original_loss": original_loss,
            "loss_loc": original_loss_loc,
            "loss_cls": original_loss_cls,
            "loss_obj": original_loss_obj,
            "loss_noo": original_loss_noo,
            'np': obj_mask.sum(),
            "lambda_loc": self.lambda_loc,
            "lambda_cls": self.lambda_cls,
            "lambda_obj": self.lambda_obj,
        }

    def _fill_target(self, y_true, b, k, x, y, c, batch_target, t):
        """辅助函数：填充目标值"""
        y_true[b, k, x, y, 0] = batch_target[t, 0] - x.float()
        y_true[b, k, x, y, 1] = batch_target[t, 1] - y.float()
        y_true[b, k, x, y, 2] = batch_target[t, 2]
        y_true[b, k, x, y, 3] = batch_target[t, 3]
        y_true[b, k, x, y, 4] = 1
        y_true[b, k, x, y, 5 + c] = 1
        return y_true

    def build_target(self, i, prediction, targets):
        B = prediction.shape[0]
        S = prediction.shape[2]
        y_true = torch.zeros_like(prediction)

        for b in range(B): # 遍历每张图片
            if targets[b].shape[0] == 0:  # 处理无目标的情况
                continue

            # 预处理目标框
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, 0:4] = targets[b][:, 0:4] * S
            batch_target[:, 4] = targets[b][:, 4]

            # 计算IOU矩阵
            gt_box = targets[b][:, 2:4]
            iou_matrix = self.compute_iou(gt_box, self.anchors, iou_type='iou')
            best_iou, best_na = torch.max(iou_matrix, dim=1)

            # 遍历每个物体
            for t, n_a in enumerate(best_na.tolist()):
                # 获取坐标和类别
                x = batch_target[t, 0].long().clamp(0, S - 1)
                y = batch_target[t, 1].long().clamp(0, S - 1)
                c = batch_target[t, 4].long()
                if n_a in self.am[i]:

                    # 处理最佳匹配的anchor
                    k = self.am[i].index(n_a)
                    y_true = self._fill_target(y_true, b, k, x, y, c, batch_target, t)

                    # 处理其他匹配的anchor
                    additional_anchors = (iou_matrix[t].split(split_size=self.na)[i] > 0.5).nonzero().squeeze(1)
                    for a in additional_anchors.tolist():
                        if a != k:
                            y_true = self._fill_target(y_true, b, a, x, y, c, batch_target, t)

                    if best_iou[t] < 0.9:  # 如果 IOU 大于阈值，则将目标框扩展到邻近格子
                        continue

                    # 计算偏移方向
                    offsets = []
                    if batch_target[t, 0] % 1 > 0.5:  # x > 0.5时右扩展
                        offsets.append((1, 0))  # 向右扩展
                    elif batch_target[t, 0] % 1 < 0.5:  # x < 0.5时左扩展
                        offsets.append((-1, 0))  # 向左扩展

                    if batch_target[t, 1] % 1 > 0.5:  # y > 0.5时下扩展
                        offsets.append((0, 1))  # 向下扩展
                    elif batch_target[t, 1] % 1 < 0.5:  # y < 0.5时上扩展
                        offsets.append((0, -1))  # 向上扩展

                    for dx, dy in offsets:
                        nx = torch.clamp(x + dx, 0, S - 1)
                        ny = torch.clamp(y + dy, 0, S - 1)

                        # 确定该邻近格子是否也匹配到同样的 anchor（根据 IOU 阈值）
                        y_true = self._fill_target(y_true, b, k, nx, ny, c, batch_target, t)
                else:
                    # 处理其他匹配的anchor
                    additional_anchors = (iou_matrix[t].split(split_size=self.na)[i] > 0.5).nonzero().squeeze(1)
                    for a in additional_anchors.tolist():
                        y_true = self._fill_target(y_true, b, a, x, y, c, batch_target, t)

        return y_true

    def ignore_target(self, i, prediction, y_true, obj_mask):
        pred_box, targ_box = None, None
        #============================================#
        #   batch size
        #============================================#
        B = prediction.shape[0]
        #============================================#
        #   特征图大小
        #============================================#
        S = prediction.shape[2]
        #============================================#
        #   构建noobj mask
        #============================================#
        noobj_mask = torch.ones_like(obj_mask, dtype=torch.bool).to(self.device)
        #============================================#
        #   遍历每一个正样本
        #   在正样本位置上的anchors 都置为0
        #============================================#
        for bs in range(B):
            if obj_mask[bs].sum().item() == 0:
                continue
            #============================================#
            #   正样本数量
            #============================================#
            N = obj_mask[bs].sum().item()
            for n in range(N):
                #============================================#
                #   获取正样本的x坐标
                #============================================#
                x = obj_mask[bs].nonzero()[n][-2:][0].item()
                #============================================#
                #   获取正样本的y坐标
                #============================================#
                y = obj_mask[bs].nonzero()[n][-2:][1].item()
                #============================================#
                #   置正样本所有anchor为0
                #============================================#
                noobj_mask[bs, :, x, y] = False
        if obj_mask.sum().item() != 0:
            #============================================#
            #   获取best anchor index
            #============================================#
            b, a, gx, gy = obj_mask.nonzero(as_tuple=True)
            anchors = self.anchors[i][a]
            #============================================#
            #   xywh 转 xyxy并动态限制tw和th
            #============================================#
            p_x = prediction[b, a, gx, gy][:, 0].sigmoid() * 2 - 0.5 + gx
            p_y = prediction[b, a, gx, gy][:, 1].sigmoid() * 2 - 0.5 + gy
            p_w = (prediction[b, a, gx, gy][:, 2].exp() * anchors[:, 0]).clamp(0, prediction.shape[2])
            p_h = (prediction[b, a, gx, gy][:, 3].exp() * anchors[:, 1]).clamp(0, prediction.shape[3])

            t_x = y_true[b, a, gx, gy][:, 0] + gx
            t_y = y_true[b, a, gx, gy][:, 1] + gy
            t_w = y_true[b, a, gx, gy][:, 2]
            t_h = y_true[b, a, gx, gy][:, 3]

            p_x1 = p_x - p_w / 2
            p_y1 = p_y - p_h / 2
            p_x2 = p_x + p_w / 2
            p_y2 = p_y + p_h / 2

            t_x1 = t_x - t_w / 2
            t_y1 = t_y - t_h / 2
            t_x2 = t_x + t_w / 2
            t_y2 = t_y + t_h / 2

            pred_box = torch.stack([p_x1, p_y1, p_x2, p_y2], dim=-1)
            targ_box = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1)

        return noobj_mask, pred_box, targ_box

    def compute_iou(self, box_1, box_2, iou_type='iou'):
        if iou_type=='iou':
            ious = []
            for i, s in enumerate(self._S):
                gt_wh = (box_1 * s).unsqueeze(1)  # [N, 1, 2]
                anchors_wh = box_2[i].unsqueeze(0)  # [1, A, 2]

                # 计算交集的宽和高
                inter_w = torch.min(gt_wh[..., 0], anchors_wh[..., 0])  # 交集宽度
                inter_h = torch.min(gt_wh[..., 1], anchors_wh[..., 1])  # 交集高度

                # 计算交集面积
                intersection = inter_w * inter_h

                # 计算每个框的面积
                gt_area = gt_wh[..., 0] * gt_wh[..., 1]  # [N, 1]的面积
                anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]  # [1, A]的面积

                # 计算并集面积
                union = gt_area + anchor_area - intersection

                iou = intersection / union

                ious.append(iou)

            return torch.cat(ious, dim=1)
        
        else:
            p_x1 = box_1[:, 0]
            p_y1 = box_1[:, 1]
            p_x2 = box_1[:, 2]
            p_y2 = box_1[:, 3]

            t_x1 = box_2[:, 0]
            t_y1 = box_2[:, 1]
            t_x2 = box_2[:, 2]
            t_y2 = box_2[:, 3]
            # 计算交集区域面积
            area_x1 = torch.max(p_x1, t_x1)
            area_y1 = torch.max(p_y1, t_y1)
            area_x2 = torch.min(p_x2, t_x2)
            area_y2 = torch.min(p_y2, t_y2)

            inter = (area_x2 - area_x1).clamp(min=0) * (area_y2 - area_y1).clamp(min=0)
            area_1 = (p_x2 - p_x1) * (p_y2 - p_y1)
            area_2 = (t_x2 - t_x1) * (t_y2 - t_y1)
            union = (area_1 + area_2 - inter).clamp(min=1e-9)

            iou = inter / union

            if iou_type == 'giou':
                # 最小包围框
                xc1 = torch.min(p_x1, t_x1)
                yc1 = torch.min(p_y1, t_y1)
                xc2 = torch.max(p_x2, t_x2)
                yc2 = torch.max(p_y2, t_y2)
                area_c = ((xc2 - xc1) * (yc2 - yc1)).clamp(min=1e-6) + 1e-16

                giou = iou - (area_c - union) / (area_c)

                return giou
        
            elif iou_type=='diou':
                # 中心点
                b1_center_x = (p_x1 + p_x2) / 2
                b1_center_y = (p_y1 + p_y2) / 2
                b2_center_x = (t_x1 + t_x2) / 2
                b2_center_y = (t_y1 + t_y2) / 2

                center_dist_sq = (b1_center_x - b2_center_x) ** 2 + (b1_center_y - b2_center_y) ** 2

                # 包裹框的对角线距离
                enclose_x1 = torch.min(p_x1, t_x1)
                enclose_y1 = torch.min(p_y1, t_y1)
                enclose_x2 = torch.max(p_x2, t_x2)
                enclose_y2 = torch.max(p_y2, p_y2)
                enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

                diou = iou - center_dist_sq / enclose_diag_sq

                return diou
        
            elif iou_type=='ciou':
                # 中心点坐标
                p_cx = (p_x1 + p_x2) / 2
                p_cy = (p_y1 + p_y2) / 2
                t_cx = (t_x1 + t_x2) / 2
                t_cy = (t_y1 + t_y2) / 2

                # 中心点距离的平方
                center_dist = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

                # 包裹框的对角线平方
                enclose_x1 = torch.min(p_x1, t_x1)
                enclose_y1 = torch.min(p_y1, t_y1)
                enclose_x2 = torch.max(p_x2, t_x2)
                enclose_y2 = torch.max(p_y2, t_y2)
                enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

                # 宽高
                p_w = (p_x2 - p_x1).clamp(min=1e-9)
                p_h = (p_y2 - p_y1).clamp(min=1e-9)
                t_w = (t_x2 - t_x1).clamp(min=1e-9)
                t_h = (t_y2 - t_y1).clamp(min=1e-9)

                # 宽高比项 v 和权重 alpha
                v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(t_w / t_h) - torch.atan(p_w / p_h), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-7)

                # CIoU
                ciou = iou - center_dist / enclose_diag - alpha * v

                return ciou
            else:
                raise ValueError(f'Invalid iou_type: {iou_type}')
            

def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    # 计算交叉熵
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    
    # 2) 计算 p_t（模型对当前标签的置信度）
    p = torch.sigmoid(pred)
    p_t = target * p + (1 - target) * (1 - p)

    # 3) 计算 α_t（正负样本不同权重）
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    
    # 4) 焦点因子
    focal_factor = (1 - p_t) ** gamma
    
    # 5) 最终 Focal Loss
    loss = alpha_t * focal_factor * bce_loss

    # 返回平均损失
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# bbox debug
# import cv2 as cv
# S = y_true.shape[2]
# S = 416 / S

# x1 = p_x1 * S
# x2 = p_x2 * S
# y1 = p_y1 * S
# y2 = p_y2 * S

# for i, image in enumerate(self.rImages):
#     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#     for j in (i == b).nonzero().squeeze(1):
#         cv.rectangle(image, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), (0, 255, 0), 2)
#     cv.imshow("image", image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()