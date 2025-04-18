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

            noobj_mask, obj_mask = self.build_target(targ, anchors)

            # no object loss
            loss += self.BCEloss(pred[..., 4][noobj_mask], targ[..., 4][noobj_mask])

            # object loss
            x = pred[..., 0][obj_mask]
            y = pred[..., 1][obj_mask]
            w = pred[..., 2][obj_mask]
            h = pred[..., 3][obj_mask]
            c = pred[..., 4][obj_mask]
            _cls = pred[..., 5:][obj_mask]

            t_x = targ[..., 0][obj_mask]
            t_y = targ[..., 1][obj_mask]
            t_w = targ[..., 2][obj_mask]
            t_h = targ[..., 3][obj_mask]
            t_c = targ[..., 4][obj_mask]
            t_cls = targ[..., 5:][obj_mask]

            x_loss = self.BCEloss(x, t_x)
            y_loss = self.BCEloss(y, t_y)
            w_loss = self.MSEloss(w, t_w)
            h_loss = self.MSEloss(h, t_h)

            loss_loc = (x_loss + y_loss + w_loss + h_loss) * 0.1
            loss_cof = self.BCEloss(c, t_c) * 1
            loss_cls = self.BCEloss(_cls, t_cls) * 0.1

            loss += loss_loc + loss_cof + loss_cls
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

    def build_target(self, target, anchors, thre=0.4):
        anchors = torch.tensor(anchors, device=self.device) / self.IMG_SIZE
        gt_wh = torch.minimum(target[..., 2:4], anchors.view(3, 1, 1, 2))

        iou_i = gt_wh[..., 0] * gt_wh[..., 1]
        iou_u = (torch.prod(target[..., 2:4], dim=-1) + (torch.prod(anchors, dim=-1)).view(3, 1, 1)) - iou_i
        iou = iou_i / iou_u

        best_iou, best_anchor_idx = torch.max(iou, dim=1)
        mask = torch.zeros_like(iou, device=target.device)  # (B, A, H, W)
        mask.scatter_(1, best_anchor_idx.unsqueeze(1), best_iou.unsqueeze(1))

        obj_mask = mask > thre
        noobj_mask = ~obj_mask

        return noobj_mask, obj_mask