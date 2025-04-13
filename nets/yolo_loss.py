import torch
import torch.nn as nn
import numpy as np
from utils.tools import iou
from config.yolov3 import CONF

class YOLOv3LOSS():
    def __init__(self, img_size, anchors: list):
        super(YOLOv3LOSS,self).__init__()
        self.device = CONF.device
        self.feature_map = CONF.feature_map
        self.IMG_SIZE = img_size
        self.anchors  = anchors

        self.BCEloss = nn.BCELoss()
        self.MSEloss = nn.MSELoss()
        pass
    def __call__(self, predict, target):
        loss = torch.zeros(1, device=self.device)
        # feats = predict[1] if isinstance(predict, tuple) else predict

        for i in range(3):
            pred = predict[i].view(-1, 3, self.feature_map[i], self.feature_map[i], 80+5)
            targ = target[i]

            pred[..., 0] = torch.sigmoid(pred[..., 0])
            pred[..., 1] = torch.sigmoid(pred[..., 0])
            pred[..., 4] = torch.sigmoid(pred[..., 0])
            pred[..., 5:] = torch.sigmoid(pred[..., 5:])

            anchors = self.anchors[i]

            noobj_mask = targ[..., 4]==0
            noobj_index = noobj_mask.nonzero()
            obj_mask = targ[..., 4]==1
            obj_index = obj_mask.nonzero()

            # no object loss
            loss += self.BCEloss(pred[..., 4][noobj_mask], targ[..., 4][noobj_mask])
            
            # obj loss
            best_iou,iou_index = self.build_target(targ, anchors)

            x_index = iou_index[:, 0]
            y_index = iou_index[:, 1]
            w_index = iou_index[:, 2]
            h_index = iou_index[:, 3]

            x = pred[x_index, y_index, w_index, h_index, 0]
            y = pred[x_index, y_index, w_index, h_index, 1]
            w = pred[x_index, y_index, w_index, h_index, 2]
            h = pred[x_index, y_index, w_index, h_index, 3]
            c = pred[x_index, y_index, w_index, h_index, 4]
            _cls = pred[x_index, y_index, w_index, h_index, 5:]

            t_x = targ[x_index, y_index, w_index, h_index, 0]
            t_y = targ[x_index, y_index, w_index, h_index, 1]
            t_w = targ[x_index, y_index, w_index, h_index, 2]
            t_h = targ[x_index, y_index, w_index, h_index, 3]
            t_c = targ[x_index, y_index, w_index, h_index, 4]
            t_cls = targ[x_index, y_index, w_index, h_index, 5:]

            x_loss = self.BCEloss(x, t_x)
            y_loss = self.BCEloss(y, t_y)
            w_loss = self.MSEloss(w, t_w)
            h_loss = self.MSEloss(h, t_h)

            loss_loc = (x_loss + y_loss + w_loss + h_loss) * 0.1
            loss_cof = self.BCEloss(c, t_c) * 1
            loss_cls = self.BCEloss(_cls, t_cls) * 0.1

            loss = loss + loss_loc + loss_cof + loss_cls
        return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)

    def build_target(self, target, anchors, iou_thre=0.5):
        anchors = torch.tensor(anchors, device=self.device) / self.IMG_SIZE

        area_w_h = torch.minimum(target[..., 2:4], anchors.view(1, 3, 1, 1, 2))

        iou_i = area_w_h[..., 0] * area_w_h[..., 1]
        iou_u = (target[..., 2:4][..., 0] * target[..., 2:4][..., 1] + (anchors[:, 0] * anchors[:, 1]).view(1, 3, 1, 1)) - iou_i
            
        iou = iou_i / iou_u
        iou_index = (iou>iou_thre).nonzero()

        return iou[iou_index[:, 0], iou_index[:, 1], iou_index[:, 2], iou_index[:, 3]], iou_index