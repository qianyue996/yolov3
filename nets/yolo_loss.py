import torch
import torch.nn as nn
import math

from config.yolov3 import CONF

class YOLOv3LOSS():
    def __init__(self):
        super(YOLOv3LOSS,self).__init__()
        self.device = CONF.device
        self.feature_map = CONF.feature_map
        self.IMG_SIZE = CONF.imgsize
        self.anchors  = CONF.anchors
        self.anchor_num = CONF.per_feat_anc_num
        self.classes_num = 80

        self.BCEloss = nn.BCELoss()
        self.MSEloss = nn.MSELoss()
        pass
    def __call__(self, predict, targets):
        loss = torch.zeros(1, device=self.device)

        for i in range(3):
            S = self.feature_map[i]

            prediction = predict[i].view(-1, 3, 5 + 80, S, S).permute(0, 1, 3, 4, 2)

            anchors = self.anchors[i]

            target = self.build_target(i, targets, anchors, S)

            obj_mask = target[..., 4] == 1

            n = obj_mask.sum()
            if n != 0:
                x = torch.sigmoid(prediction[..., 0][obj_mask])
                t_x = target[..., 0][obj_mask]

                y = torch.sigmoid(prediction[..., 1][obj_mask])
                t_y = target[..., 1][obj_mask]

                w = prediction[..., 2][obj_mask]
                t_w = target[..., 2][obj_mask]

                h = prediction[..., 3][obj_mask]
                t_h = target[..., 3][obj_mask]

                c = torch.sigmoid(prediction[..., 4][obj_mask])
                t_c = target[..., 4][obj_mask]

                _cls = torch.sigmoid(prediction[..., 5:][obj_mask])
                t_cls = target[..., 5:][obj_mask]

                # noobj target
                noobj_mask = ~obj_mask
                no_conf = torch.sigmoid(prediction[..., 4][noobj_mask])
                no_t_conf = torch.zeros_like(no_conf)
                loss += self.BCEloss(no_conf, no_t_conf)

                loss_x = self.BCEloss(x, t_x)
                loss_y = self.BCEloss(y, t_y)
                loss_w = self.MSEloss(w, t_w)
                loss_h = self.MSEloss(h, t_h)
                loss_loc = (loss_x + loss_y + loss_w + loss_h) * 1

                loss_cls = self.BCEloss(_cls, t_cls) * 2
                loss_conf = self.BCEloss(c, t_c)
                
                loss += loss_loc + loss_cls + loss_conf

        return loss

    def build_target(self, i, targets, anchors, S, thre=0.4):
        B = len(targets)
        target = torch.zeros(B, 3, S, S, 5 + 80, device=self.device)

        for bs in range(B):
            batch_target = targets[bs].clone()
            batch_target[:, 0:2] = targets[bs][:, 0:2] * S
            batch_target[:, 2:4] = targets[bs][:, 2:4]
            batch_target[:, 4] = targets[bs][:, 4]

            gt_box = batch_target[:, 2:4] * self.IMG_SIZE

            _anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)

            best_iou, best_na = torch.max(self.compute_iou(gt_box, _anchors), dim=1)

            for index, n_a in enumerate(best_na):
                if best_iou[index] < thre:
                    continue

                k = n_a

                x = batch_target[index, 0].long()

                y = batch_target[index, 1].long()

                c = batch_target[index, 4].long()

                target[bs, k, x, y, 0] = batch_target[index, 0] - x.float()
                target[bs, k, x, y, 1] = batch_target[index, 1] - y.float()
                target[bs, k, x, y, 2] = torch.log(batch_target[index, 2] / _anchors[k][0])
                target[bs, k, x, y, 3] = torch.log(batch_target[index, 3] / _anchors[k][1])
                target[bs, k, x, y, 4] = 1
                target[bs, k, x, y, 5 + c] = 1

        return target
    def compute_iou(self, gt_box, anchors):
        gt_box = gt_box.unsqueeze(1)
        anchors = anchors.unsqueeze(0)

        min_wh = torch.min(gt_box, anchors)
        iou_i = min_wh[..., 0] * min_wh[..., 1]

        area_a = (gt_box[..., 0] * gt_box[..., 1]).expand_as(iou_i)
        area_b =  (anchors[..., 0] * anchors[..., 1]).expand_as(iou_i)
        iou_u = area_a + area_b - iou_i

        return iou_i / iou_u