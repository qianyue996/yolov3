import torch
import torch.nn as nn
import numpy as np
from utils.tools import iou
from config.yolov3 import CONF

class YOLOv3LOSS():
    def __init__(self):
        super(YOLOv3LOSS,self).__init__()
        self.device = CONF.device
        self.feature_map = CONF.feature_map
        self.IMG_SIZE = CONF.imgsize
        self.anchors  = CONF.anchors
        self.anchor_num = CONF.per_feat_anc_num
        self.classes_num = CONF.classNumber

        self.BCEloss = nn.BCELoss()
        self.MSEloss = nn.MSELoss()
        pass
    def __call__(self, predict, target):
        loss = torch.zeros(1, device=self.device)

        for i in range(3):
            pred = predict[i].view(-1, self.anchor_num,
                                   5 + self.classes_num,
                                   self.feature_map[i],
                                   self.feature_map[i]).permute(0, 1, 3, 4, 2)
            targ = target[i]

            pred[..., 0] = torch.sigmoid(pred[..., 0])
            pred[..., 1] = torch.sigmoid(pred[..., 0])
            pred[..., 4] = torch.sigmoid(pred[..., 0])
            pred[..., 5:] = torch.sigmoid(pred[..., 5:])

            anchors = self.anchors[i]

            for bs in range(pred.shape[0]):
                one_pred = pred[bs]
                one_targ = targ[bs]

                noobj_mask, obj_mask = self.build_target(targ[bs], anchors)

                # no object loss
                loss += self.BCEloss(one_pred[..., 4][noobj_mask], one_targ[..., 4][noobj_mask])

                x = one_pred[..., 0][obj_mask]
                y = one_pred[..., 1][obj_mask]
                w = one_pred[..., 2][obj_mask]
                h = one_pred[..., 3][obj_mask]
                c = one_pred[..., 4][obj_mask]
                _cls = one_pred[..., 5:][obj_mask]

                t_x = one_targ[..., 0][obj_mask]
                t_y = one_targ[..., 1][obj_mask]
                t_w = one_targ[..., 2][obj_mask]
                t_h = one_targ[..., 3][obj_mask]
                t_c = one_targ[..., 4][obj_mask]
                t_cls = one_targ[..., 5:][obj_mask]

                x_loss = self.BCEloss(x, t_x)
                y_loss = self.BCEloss(y, t_y)
                w_loss = self.MSEloss(w, t_w)
                h_loss = self.MSEloss(h, t_h)

                loss_loc = (x_loss + y_loss + w_loss + h_loss) * 0.1
                loss_cof = self.BCEloss(c, t_c) * 1
                loss_cls = self.BCEloss(_cls, t_cls) * 0.1

                loss = loss + loss_loc + loss_cof + loss_cls
        return loss

    def build_target(self, target, anchors):
        anchors = torch.tensor(anchors, device=self.device) / self.IMG_SIZE
        gt_wh = torch.minimum(target[..., 2:4], anchors.view(3, 1, 1, 2))

        iou_i = gt_wh[..., 0] * gt_wh[..., 1]
        iou_u = (torch.prod(target[..., 2:4], dim=-1) + (torch.prod(anchors, dim=-1)).view(3, 1, 1)) - iou_i
        iou = iou_i / iou_u

        max_iou = iou.max() # 取最好的anchor
        iou_index = (iou == max_iou).nonzero().squeeze()
        # best_iou_index = iou_index.nonzero().squeeze()
        # best_anchor_num = torch.where(iou == max_iou.max())[0].item()

        obj_mask = torch.zeros_like(iou, dtype=torch.bool)

        obj_mask[iou_index[0], iou_index[1], iou_index[2]] = True
        noobj_mask = ~obj_mask
       
        return noobj_mask, obj_mask