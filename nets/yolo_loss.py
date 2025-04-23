import torch

from config.yolov3 import CONF

class YOLOv3LOSS():
    def __init__(self):
        super(YOLOv3LOSS,self).__init__()
        self.device = CONF.device
        self.feature_map = CONF.feature_map
        self.IMG_SIZE = CONF.imgsize
        self.anchors  = CONF.anchors
        self.classes_num = 80
    def BCELoss(self, x, y):
        eps = 1e-7
        x = torch.clamp(x, eps, 1 - eps)
        return - (y * torch.log(x) + (1 - y) * torch.log(1 - x))
    
    def MSELoss(self, x, y):
        return (x - y) ** 2
    def __call__(self, predict, targets, writer, global_step):
        loss = torch.zeros(1, device=self.device)

        for i in range(3):
            S = self.feature_map[i]
            B = predict[i].shape[0]
            stride = CONF.sample_ratio[i]

            prediction = predict[i].view(-1, 3, 5 + 80, S, S).permute(0, 1, 3, 4, 2)

            anchors = torch.LongTensor(self.anchors[i], device=self.device)

            target, ignore_target, i_j, b_n = self.build_target(i, targets, anchors, S)

            obj_mask = target[..., 4] == 1
            ignore_mask = ignore_target[..., 4] == 1
            noobj_mask = ~ignore_mask

            n = obj_mask.sum().item()
            if n != 0:
                i_x = i_j[..., 0][obj_mask]
                i_y = i_j[..., 1][obj_mask]

                x = (torch.sigmoid(prediction[..., 0])[obj_mask] + i_x) * stride
                t_x = target[..., 0][obj_mask] + i_j[..., 0][obj_mask] * stride

                y = (torch.sigmoid(prediction[..., 1][obj_mask]) + i_y) * stride
                t_y = target[..., 1][obj_mask] + i_j[..., 0][obj_mask] * stride

                w = torch.exp(prediction[..., 2][obj_mask]) * anchors[b_n.long(), 0]
                t_w = torch.exp(target[..., 2][obj_mask]) * anchors[b_n.long(), 0]

                h = torch.exp(prediction[..., 3][obj_mask]) * anchors[b_n.long(), 1]
                t_h = torch.exp(target[..., 3][obj_mask]) * anchors[b_n.long(), 1]

                c = torch.sigmoid(prediction[..., 4][obj_mask])
                t_c = target[..., 4][obj_mask]

                _cls = torch.sigmoid(prediction[..., 5:][obj_mask])
                t_cls = target[..., 5:][obj_mask]

                # noobj target
                no_conf = torch.sigmoid(prediction[..., 4][noobj_mask])
                no_t_conf = torch.zeros_like(no_conf)
                noobj_loss = self.BCELoss(no_conf, no_t_conf).mean()
                
                # giou
                x1 = x - w / 2
                t_x1 = t_x - t_w / 2

                y1 = y - h / 2
                t_y1 = t_y - t_h / 2

                x2 = x + w / 2
                t_x2 = t_x + t_w / 2

                y2 = y + h / 2
                t_y2 = t_y + t_h / 2

                gt_box = torch.stack([x1, y1, x2, y2], dim=1)
                t_gt_box = torch.stack([t_x1, t_y1, t_x2, t_y2], dim=1)
                giou = self.compute_giou(gt_box, t_gt_box)

                loss_loc = (1 - giou).mean()
                loss_conf = self.BCELoss(c, t_c).mean()
                loss_cls = self.BCELoss(_cls, t_cls).mean(dim=0).sum()
                
                loss += noobj_loss + loss_loc + loss_conf + loss_cls

                writer.add_scalar(f'noobj_feture: {S}', noobj_loss, global_step)
                writer.add_scalar(f'xywh_feture: {S}', loss_loc, global_step)
                writer.add_scalar(f'conf_feture: {S}', loss_conf, global_step)
                writer.add_scalar(f'cls_feture: {S}', loss_cls, global_step)

        return loss

    def build_target(self, i, targets, anchors, S, thre=0.4):
        B = len(targets)
        target = torch.zeros(B, 3, S, S, 5 + 80, device=self.device)
        ignore_target = target.clone() # 忽略非最好的两个anchors
        i_j = target.clone()
        b_n = []

        for bs in range(B):
            batch_target = targets[bs].clone()

            batch_target[:, 0:2] = targets[bs][:, 0:2] * S
            batch_target[:, 2:4] = targets[bs][:, 2:4]
            batch_target[:, 4] = targets[bs][:, 4]

            gt_box = batch_target[:, 2:4] * self.IMG_SIZE

            best_iou, best_na = torch.max(self.compute_iou(gt_box, anchors), dim=1)

            for index, n_a in enumerate(best_na):
                if best_iou[index] < thre:
                    continue

                k = n_a

                x = torch.floor(batch_target[index, 0]).long()

                y = torch.floor(batch_target[index, 1]).long()

                c = batch_target[index, 4].long()

                target[bs, k, x, y, 0] = batch_target[index, 0] - x.float()
                target[bs, k, x, y, 1] = batch_target[index, 1] - y.float()
                target[bs, k, x, y, 2] = torch.log(batch_target[index, 2]) / anchors[k][0]
                target[bs, k, x, y, 3] = torch.log(batch_target[index, 3]) / anchors[k][1]
                target[bs, k, x, y, 4] = 1
                target[bs, k, x, y, 5 + c] = 1

                i_j[bs, k, x, y, 0] = x.float()
                i_j[bs, k, x, y, 1] = y.float()

                b_n.append(k)

                ignore_target[bs, :, x, y, 4] = 1 # 将对应grid cell的conf设置为1，剩下的grid cell就是背景

        return target, ignore_target, i_j, torch.LongTensor(b_n, device=self.device)
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
