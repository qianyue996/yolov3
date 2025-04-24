import torch

from config.yolov3 import CONF
from utils.tools import compute_iou, clear

class YOLOv3LOSS():
    def __init__(self):
        super(YOLOv3LOSS,self).__init__()
        self.device = CONF.device
        self.feature_map = CONF.feature_map
        self.IMG_SIZE = CONF.imgsize
        self.anchors  = CONF.anchors
        self.anchors_mask = CONF.anchors_mask
        self.classes_num = len(CONF.class_name)
        self.stride = CONF.sample_ratio

        self.conf_lambda = [0.4, 1.0, 4]
        self.loc_lambda = 0.05
        self.obj_lambda = 5
        self.noobj_lambda = 0.5
        self.cls_lambda = 1
        self.eps = 1e-16
    def BCELoss(self, x, y):
        eps = 1e-7
        x = torch.clamp(x, eps, 1 - eps)
        return - (y * torch.log(x) + (1 - y) * torch.log(1 - x))
    
    def MSELoss(self, x, y):
        return (x - y) ** 2
    def __call__(self, predict, targets):
        loss_loc = torch.zeros(1, device=self.device)
        obj_conf = loss_loc.clone()
        noobj_conf = loss_loc.clone()
        loss_cls = loss_loc.clone()

        for i in range(3):
            B = predict[i].shape[0]
            S = self.feature_map[i]
            stride = self.stride[i]
            
            anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)

            prediction = predict[i].view(-1, 3, 5 + 80, S, S).permute(0, 1, 3, 4, 2)

            y_true = self.build_target(i, B, S, targets, anchors)

            x = torch.sigmoid(prediction[..., 0])

            y = torch.sigmoid(prediction[..., 1])

            w = prediction[..., 2]

            h = prediction[..., 3]

            conf = torch.sigmoid(prediction[..., 4])
            t_conf = y_true[..., 4]

            obj_mask = y_true[..., 4] == 1
            noobj_mask = ~obj_mask
            
            _cls = torch.sigmoid(prediction[..., 5:])[obj_mask]
            t_cls = y_true[..., 5:][obj_mask]

            giou = self.new_function(x, y, w, h, obj_mask, anchors, stride, y_true)
            loss_loc = (1 - giou).mean() * self.loc_lambda

            loss_cls = self.BCELoss(_cls, t_cls).mean() * self.cls_lambda
            
            loss_conf = self.BCELoss(conf, t_conf)
            obj_conf = loss_conf[obj_mask].mean() * self.conf_lambda[i] * self.obj_lambda
            noobj_conf = loss_conf[noobj_mask].mean() * self.conf_lambda[i] * self.noobj_lambda

            loss_loc += loss_loc
            obj_conf += obj_conf
            noobj_conf += noobj_conf
            loss_cls = loss_cls
        
        loss = loss_loc + obj_conf + noobj_conf + loss_cls

        return {'loss':loss,
                'loss_loc':loss_loc,
                'obj_conf': obj_conf,
                'noobj_conf': noobj_conf,
                'loss_cls': loss_cls}

    def build_target(self, i, B, S, targets, anchors):
        y_true = torch.zeros(B, 3, S, S, 5 + 80, device=self.device)

        for bs in range(B):
            if targets[bs].shape[0] == 0:  # 处理无目标的情况
                continue

            batch_target = torch.zeros_like(targets[bs])
            batch_target[:, 0:2] = targets[bs][:, 0:2] * S
            batch_target[:, 2:4] = targets[bs][:, 2:4] * self.IMG_SIZE
            batch_target[:, 4] = targets[bs][:, 4]

            gt_box = batch_target[:, 2:4]
            best_iou, best_na = torch.max(self.compute_iou(gt_box, anchors), dim=1)

            for index, n_a in enumerate(best_na.tolist()):
                if n_a not in self.anchors_mask[i]:
                    continue

                k = self.anchors_mask[i].index(n_a)
                x = torch.clamp(batch_target[index, 0].long(), 0, S-1)
                y = torch.clamp(batch_target[index, 1].long(), 0, S-1)
                c = batch_target[index, 4].long()

                y_true[bs, k, x, y, 0] = batch_target[index, 0] - x.float()
                y_true[bs, k, x, y, 1] = batch_target[index, 1] - y.float()
                y_true[bs, k, x, y, 2] = torch.log(batch_target[index, 2] / anchors[k][0])
                y_true[bs, k, x, y, 3] = torch.log(batch_target[index, 3] / anchors[k][1])
                y_true[bs, k, x, y, 4] = 1
                y_true[bs, k, x, y, 5 + c] = 1

        return y_true
    def compute_iou(self, gt_box, anchors):
        gt_box = gt_box.unsqueeze(1)
        anchors = anchors.unsqueeze(0)

        min_wh = torch.min(gt_box, anchors)
        iou_i = min_wh[..., 0] * min_wh[..., 1]

        area_a = (gt_box[..., 0] * gt_box[..., 1]).expand_as(iou_i)
        area_b =  (anchors[..., 0] * anchors[..., 1]).expand_as(iou_i)
        iou_u = area_a + area_b - iou_i

        return iou_i / iou_u
    
    def compute_giou(self, gt_box, t_gt_box):
        # box1, box2: [N, 4], in (x1, y1, x2, y2)
        x1 = torch.max(gt_box[:, 0], t_gt_box[:, 0])
        y1 = torch.max(gt_box[:, 1], t_gt_box[:, 1])
        x2 = torch.min(gt_box[:, 2], t_gt_box[:, 2])
        y2 = torch.min(gt_box[:, 3], t_gt_box[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
        area2 = (t_gt_box[:, 2] - t_gt_box[:, 0]) * (t_gt_box[:, 3] - t_gt_box[:, 1])
        union = area1 + area2 - inter

        iou = inter / union.clamp(min=1e-6)

        # 最小包围框
        xc1 = torch.min(gt_box[:, 0], t_gt_box[:, 0])
        yc1 = torch.min(gt_box[:, 1], t_gt_box[:, 1])
        xc2 = torch.max(gt_box[:, 2], t_gt_box[:, 2])
        yc2 = torch.max(gt_box[:, 3], t_gt_box[:, 3])
        area_c = (xc2 - xc1) * (yc2 - yc1)

        giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
        return giou

    def new_function(self, x, y, w, h, obj_mask, anchors, stride, y_true):
        best_a = obj_mask.nonzero()[:, 1]
        grid_x = obj_mask.nonzero()[:, 2]
        grid_y = obj_mask.nonzero()[:, 3]

        x = (x[obj_mask] + grid_x) * stride
        y = (y[obj_mask] + grid_y) * stride
        w = torch.exp(w[obj_mask]) * anchors[best_a][:, 0]
        h = torch.exp(h[obj_mask]) * anchors[best_a][:, 1]

        t_x = y_true[obj_mask][:, 0] * stride
        t_y = y_true[obj_mask][:, 1] * stride
        t_w = torch.exp(y_true[obj_mask][:, 2]) * anchors[best_a][:, 0]
        t_h = torch.exp(y_true[obj_mask][:, 3]) * anchors[best_a][:, 1]

        x1 = torch.clamp(x - w / 2, min=1e-6, max=self.IMG_SIZE)
        y1 = torch.clamp(y - h / 2, min=1e-6, max=self.IMG_SIZE)
        x2 = torch.clamp(x + w / 2, min=1e-6, max=self.IMG_SIZE)
        y2 = torch.clamp(y + h / 2, min=1e-6, max=self.IMG_SIZE)

        t_x1 = torch.clamp(t_x - t_w / 2, min=1e-6, max=self.IMG_SIZE)
        t_y1 = torch.clamp(t_y - t_h / 2, min=1e-6, max=self.IMG_SIZE)
        t_x2 = torch.clamp(t_x + t_w / 2, min=1e-6, max=self.IMG_SIZE)
        t_y2 = torch.clamp(t_y + t_h / 2, min=1e-6, max=self.IMG_SIZE)

        pred_box = torch.stack([x1, y1, x2, y2], dim=-1)
        targ_box = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=-1)

        giou = self.compute_giou(pred_box, targ_box)
        
        return giou
