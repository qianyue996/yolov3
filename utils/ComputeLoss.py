import torch
import torch.nn as nn
import numpy as np

class YOLOLOSS:
    def __init__(self, model):
        self.model = model
        self.anchors = model.model[-1].anchors
        self.number_anchors = model.model[-1].na
        self.number_layers = model.model[-1].nl
        self.stride = model.stride

        self.balance = [4.0, 1.0, 0.4]

        self.l_loc = 1
        self.l_cls = 1
        self.l_obj = 1

    def __call__(self, p, targets):
        device = p[0].device
        y_true = self.build_target(p, targets)
        noobj_mask, pred_box, targ_box = self.ignore_target(p, y_true)
        loc_loss, cls_loss, obj_loss = torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device)
        obj_mask = [(i[..., 4]==1) for i in y_true]
        for l, l_p in enumerate(p):
            num_targets = obj_mask[l].sum().item()

            if num_targets > 0:
                #============================================#
                #   位置损失
                #============================================#
                grid_size = l_p.shape[2]
                targ_w_norm = (targ_box[l][:, 2] - targ_box[l][:, 0]) / grid_size
                targ_h_norm = (targ_box[l][:, 3] - targ_box[l][:, 1]) / grid_size
                wihi = 2.0 - targ_w_norm * targ_h_norm
                
                giou = self.compute_iou(pred_box[l], targ_box[l], iou_type='ciou')
                loc_loss += ((1 - giou) * wihi).sum()
                #============================================#
                #   分类损失
                #============================================#
                p_cls = l_p[..., 5:][obj_mask[l]]
                t_cls = y_true[l][..., 5:][obj_mask[l]]
                cls_loss += nn.BCEWithLogitsLoss(reduction='sum')(p_cls, t_cls)
            #============================================#
            #   置信度损失
            #============================================#
            # 正样本：有目标的位置
            if num_targets > 0:
                p_conf_pos = l_p[..., 4][obj_mask[l]]
                t_conf_pos = y_true[l][..., 4][obj_mask[l]]
                obj_loss += nn.BCEWithLogitsLoss(reduction='sum')(p_conf_pos, t_conf_pos) * self.balance[l]

            # 负样本：无目标的位置
            if noobj_mask[l].sum() > 0:
                p_conf_neg = l_p[..., 4][noobj_mask[l]]
                t_conf_neg = torch.zeros_like(p_conf_neg)  # 负样本目标置信度为0
                obj_loss += nn.BCEWithLogitsLoss(reduction='sum')(p_conf_neg, t_conf_neg) * self.balance[l]  # 负样本权重较小
        #============================================#
        #   计算总loss
        #============================================#
        loc_loss *= self.l_loc
        cls_loss *= self.l_cls
        obj_loss *= self.l_obj

        loss = (loc_loss + cls_loss + obj_loss)

        return {
            "loss": loss,
            "loc_loss": loc_loss.item(),
            "cls_loss": cls_loss.item(),
            "obj_loss": obj_loss.item(),
        }

    def build_target(self, p, targets):
        y_list = []
        for n_layer in range(self.number_layers):
            grid_size = p[n_layer].shape[2]
            y_true = torch.zeros_like(p[n_layer])
            for target_index, target in enumerate(targets):
                gt_wh = target[:, 2:4] * grid_size
                gt_wh = gt_wh.unsqueeze(1)
                anchors = self.anchors[n_layer]
                
                # 计算IoU
                area_w = torch.min(gt_wh[..., 0], anchors[..., 0])
                area_h = torch.min(gt_wh[..., 1], anchors[..., 1])
                inter = area_w * area_h
                area_a = gt_wh[..., 0] * gt_wh[..., 1]
                area_b = anchors[..., 0] * anchors[..., 1]
                union = area_a + area_b - inter
                iou = inter / union
                max_iou, best_anchor_index = torch.max(iou, dim=1)

                for i, anchor_index in enumerate(best_anchor_index.tolist()):
                    if max_iou[i] < 0.5:
                        continue
                    
                    # 获取当前目标的完整信息
                    single_target = target[i]  # [x, y, w, h, c]
                    
                    # 将坐标转换到当前层的特征图尺度
                    scaled_target = single_target[:4] * grid_size
                    scaled_x, scaled_y, scaled_w, scaled_h = scaled_target
                    
                    # 计算网格位置
                    b = target_index
                    k = anchor_index
                    x = int(scaled_x.clamp(0, grid_size - 1).item())
                    y = int(scaled_y.clamp(0, grid_size - 1).item())
                    c = int(single_target[4].item())

                    # 设置目标值
                    y_true[b, k, x, y, 0] = scaled_x % 1  # x偏移
                    y_true[b, k, x, y, 1] = scaled_y % 1  # y偏移
                    y_true[b, k, x, y, 2] = scaled_w      # 宽度
                    y_true[b, k, x, y, 3] = scaled_h      # 高度
                    y_true[b, k, x, y, 4] = 1             # 置信度
                    y_true[b, k, x, y, 5 + c] = 1         # 类别
                    
            y_list.append(y_true)
        return y_list

    def ignore_target(self, p, y_true):
        pred_box, targ_box = [], []
        noobj_mask = []
        for n_layer in range(self.number_layers):
            obj_mask = y_true[n_layer][..., 4] == 1
            b, a, x, y = obj_mask.nonzero(as_tuple=True)
            l_mask = ~obj_mask
            noobj_mask.append(l_mask)
            
            if obj_mask.sum().item() == 0:
                pred_box.append(torch.empty(0, 4, device=p[n_layer].device))
                targ_box.append(torch.empty(0, 4, device=p[n_layer].device))
                continue
            p_x = p[n_layer][b, a, x, y, 0].sigmoid() * 2 - 0.5 + x
            p_y = p[n_layer][b, a, x, y, 1].sigmoid() * 2 - 0.5 + y
            anchor_w = self.anchors[n_layer][a, 0]
            anchor_h = self.anchors[n_layer][a, 1]
            p_w = torch.square(p[n_layer][b, a, x, y, 2].sigmoid() * 2) * anchor_w
            p_h = torch.square(p[n_layer][b, a, x, y, 3].sigmoid() * 2) * anchor_h

            t_x = y_true[n_layer][b, a, x, y, 0] + x
            t_y = y_true[n_layer][b, a, x, y, 1] + y
            t_w = y_true[n_layer][b, a, x, y, 2]
            t_h = y_true[n_layer][b, a, x, y, 3]

            p_x1 = p_x - p_w / 2
            p_y1 = p_y - p_h / 2
            p_x2 = p_x + p_w / 2
            p_y2 = p_y + p_h / 2

            t_x1 = t_x - t_w / 2
            t_y1 = t_y - t_h / 2
            t_x2 = t_x + t_w / 2
            t_y2 = t_y + t_h / 2

            pred_box.append(torch.stack([p_x1, p_y1, p_x2, p_y2], dim=-1))
            targ_box.append(torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1))

        return noobj_mask, pred_box, targ_box

    def compute_iou(self, box_1, box_2, iou_type='giou'):
        p_x1 = box_1[:, 0]
        p_y1 = box_1[:, 1]
        p_x2 = box_1[:, 2]
        p_y2 = box_1[:, 3]

        t_x1 = box_2[:, 0]
        t_y1 = box_2[:, 1]
        t_x2 = box_2[:, 2]
        t_y2 = box_2[:, 3]

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
            raise ValueError('Unsupported type of loss.')
            

def focal_loss(pred, targ, alpha=0.25, gamma=1.5, reduction='mean'):
    loss = nn.BCEWithLogitsLoss(reduction='none')(pred, targ)
    
    p = torch.sigmoid(pred)
    p_t = targ * p + (1 - targ) * (1 - p)
    alpha_factor = targ * alpha + (1 - targ) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= alpha_factor * modulating_factor

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