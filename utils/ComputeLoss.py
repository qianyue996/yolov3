import torch
import torch.nn as nn
import numpy as np

class YOLOv3LOSS:
    def __init__(self, model):
        self.anchors = model.model[-1].anchors
        self.na = self.anchors[0].shape[0]
        self.am = list(map(tuple, np.split(np.arange(self.anchors.view(-1, 2).shape[0]), self.anchors.view(-1, 2).shape[0] // self.anchors[0].shape[0])))
        self.nl = model.model[-1].nl

        self.balance = [4.0, 1.0, 0.4]

        self.l_loc = 0.25
        self.l_cls = 4
        self.l_obj = 10

    def __call__(self, p, targets):
        device = p[0].device
        y_true = self.build_target(p, targets)
        noobj_mask, pred_box, targ_box = self.ignore_target(p, y_true)
        loc_loss, cls_loss, obj_loss = torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device)
        obj_mask = [(i[..., 4]==1) for i in y_true]
        for l, l_p in enumerate(p):
            if obj_mask[l].sum().item() != 0:
                #============================================#
                #   位置损失
                #============================================#
                wihi = 2.0 - ((targ_box[l][:, 2] * targ_box[l][:, 3]) / p[l].shape[2] ** 2)
                giou = self.compute_iou(pred_box[l], targ_box[l], iou_type='ciou')
                loc_loss += ((1 - giou) * wihi).mean()
                #============================================#
                #   分类损失
                #============================================#
                _cls = p[l][..., 5:][obj_mask[l]]
                t_cls = y_true[l][..., 5:][obj_mask[l]]
                cls_loss += nn.BCEWithLogitsLoss(reduction='mean')(_cls, t_cls)
            #============================================#
            #   置信度损失
            #============================================#  
            conf = l_p[..., 4]
            t_conf = y_true[l][..., 4]
            scale_pos = obj_mask[l].sum().item() / (obj_mask[l].sum().item() + noobj_mask[l].sum().item())
            obj_loss_temp = nn.BCEWithLogitsLoss(reduction="none")(conf, t_conf) * self.balance[l]
            obj_loss += obj_loss_temp[obj_mask[l]].mean() * scale_pos
            obj_loss += obj_loss_temp[noobj_mask[l]].mean() * (1 - scale_pos)
        #============================================#
        #   没加lambda系数的loss，方便观察loss下降情况
        #============================================#
        original_loss_loc = loc_loss.clone().item()
        original_loss_cls = cls_loss.clone().item()
        original_loss_obj = obj_loss.clone().item()
        original_loss = original_loss_loc + original_loss_cls + original_loss_obj
        #============================================#
        #   计算总loss
        #============================================#

        loc_loss *= self.l_loc
        cls_loss *= self.l_cls
        obj_loss *= self.l_obj

        loss = loc_loss + cls_loss + obj_loss

        return {
            "loss": loss,
            "original_loss": original_loss,
            "loss_loc": original_loss_loc,
            "loss_cls": original_loss_cls,
            "loss_obj": original_loss_obj,
            'np': sum([i.sum().item() for i in obj_mask])
        }


    def build_target(self, p, targets):
        y_list = []
        if len(targets) != p[0].shape[0]:
            raise ValueError("targets.shape[0] != p.shape[0]")
        
        S = [i.shape[2] for i in p]
        for l in range(self.nl):
            y_true = torch.zeros_like(p[l])
            for t, target in enumerate(targets):
                if target.shape[0] == 0:
                    continue
                ious = []
                for _l in range(self.nl):
                    gt_box = (target[:, 2:4] * S[_l]).unsqueeze(1) # [N, 1, 2]
                    anchors = self.anchors[_l] # [1, na * 3, 2]

                    area_w = torch.min(gt_box[..., 0], anchors[..., 0])
                    area_h = torch.min(gt_box[..., 1], anchors[..., 1])
                    inter = area_w * area_h

                    area_a = gt_box[..., 0] * gt_box[..., 1]
                    area_b = anchors[..., 0] * anchors[..., 1]
                    union = area_a + area_b - inter

                    iou = inter / union
                    ious.append(iou)
                iou = torch.cat(ious).view(len(target), -1)

                max_iou, best_a = torch.max(iou, dim=1)

                for i, ai in enumerate(best_a.tolist()):
                    if max_iou[i] < 0.2:
                        if ai not in self.am[l]:
                            continue
                        b_targ = target[i][:4] * S[l]

                        b = t
                        k = self.am[l].index(ai)
                        x = b_targ[0].long().clamp(0, S[l] - 1).item()
                        y = b_targ[1].long().clamp(0, S[l] - 1).item()
                        c = target[i][4].long().item()

                        y_true[b, k, x, y, 0] = b_targ[0] % 1
                        y_true[b, k, x, y, 1] = b_targ[1] % 1
                        y_true[b, k, x, y, 2] = b_targ[2]
                        y_true[b, k, x, y, 3] = b_targ[3]
                        y_true[b, k, x, y, 4] = 1
                        y_true[b, k, x, y, 5 + c] = 1

                    else:
                        for j in (iou[i] >= 0.2).nonzero():
                            j = j.item()
                            if j not in self.am[l]:
                                continue

                            b_targ = target[i][:4] * S[l]

                            b = t
                            k = self.am[l].index(j)
                            x = b_targ[0].long().clamp(0, S[l] - 1).item()
                            y = b_targ[1].long().clamp(0, S[l] - 1).item()
                            c = target[i][4].long().item()

                            y_true[b, k, x, y, 0] = b_targ[0] % 1
                            y_true[b, k, x, y, 1] = b_targ[1] % 1
                            y_true[b, k, x, y, 2] = b_targ[2]
                            y_true[b, k, x, y, 3] = b_targ[3]
                            y_true[b, k, x, y, 4] = 1
                            y_true[b, k, x, y, 5 + c] = 1

                            if x <= 0 and y <= 0 or x >= S[l]-1 and y >= S[l]-1:
                                continue
                            offsets = []
                            if b_targ[0] % 1 > 0.5:  # x > 0.5时右扩展
                                offsets.append((1, 0))  # 向右扩展
                            elif b_targ[0] % 1 < 0.5:  # x < 0.5时左扩展
                                offsets.append((-1, 0))  # 向左扩展

                            if b_targ[1] % 1 > 0.5:  # y > 0.5时下扩展
                                offsets.append((0, 1))  # 向下扩展
                            elif b_targ[1] % 1 < 0.5:  # y < 0.5时上扩展
                                offsets.append((0, -1))  # 向上扩展

                            for dx, dy in offsets:
                                nx = max(0, min(S[l] - 1, x + dx))
                                ny = max(0, min(S[l] - 1, y + dy))

                                y_true[b, k, nx, ny, 0] = abs(b_targ[0] - nx)
                                y_true[b, k, nx, ny, 1] = abs(b_targ[1] - ny)
                                y_true[b, k, nx, ny, 2] = b_targ[2]
                                y_true[b, k, nx, ny, 3] = b_targ[3]
                                y_true[b, k, nx, ny, 4] = 1
                                y_true[b, k, nx, ny, 5 + c] = 1

            y_list.append(y_true)

        return y_list

    def ignore_target(self, p, y_true):
        pred_box, targ_box = [], []
        noobj_mask = []
        for l in range(self.nl):
            obj_mask = y_true[l][..., 4] == 1
            b, a, x, y = obj_mask.nonzero(as_tuple=True)
            # l_mask = torch.ones_like(obj_mask, dtype=torch.bool)
            # l_mask[b, :, x, y] = False
            l_mask = ~obj_mask
            noobj_mask.append(l_mask)
            
            if obj_mask.sum().item() == 0:
                pred_box.append(0)
                targ_box.append(0)
                continue
            p_x = p[l][b, a, x, y, 0].sigmoid() * 2 - 0.5 + x
            p_y = p[l][b, a, x, y, 1].sigmoid() * 2 - 0.5 + y
            p_w = torch.square(p[l][b, a, x, y, 2].sigmoid() * 2) * self.anchors[l][a][:, 0]
            p_h = torch.square(p[l][b, a, x, y, 3].sigmoid() * 2) * self.anchors[l][a][:, 1]

            t_x = y_true[l][b, a, x, y, 0] + x
            t_y = y_true[l][b, a, x, y, 1] + y
            t_w = y_true[l][b, a, x, y, 2]
            t_h = y_true[l][b, a, x, y, 3]

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

    def compute_iou(self, box_1=None, box_2=None, iou_type='giou'):
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