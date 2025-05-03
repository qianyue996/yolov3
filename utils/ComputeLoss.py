import torch
import torch.nn as nn


class YOLOv3LOSS:
    def __init__(self, model, device, l_loc, l_cls, l_obj, l_noo):
        self.device = device
        self.stride = model.model[-1].stride
        self.anchors = model.model[-1].anchors

        self.lambda_obj_layers = [1.0, 1.0, 1.0]

        self.lambda_loc = l_loc
        self.lambda_cls = l_cls
        self.lambda_obj = l_obj
        self.lambda_noo = l_noo

    def __call__(self, predictions, targets, item):
        all_loss_loc = torch.zeros(1).to(self.device)
        all_loss_cls = all_loss_loc.clone()
        all_loss_obj = all_loss_loc.clone()
        all_loss_noo = all_loss_loc.clone()
        # ===========================================#
        #
        # ===========================================#
        for i, prediction in enumerate(predictions):
            # ===========================================#
            #   参数
            # ===========================================#
            B = prediction.shape[0]
            # ===========================================#
            #   获取feature map的宽高
            #   将图像分割为SxS个网格
            #   13,26,52
            # ===========================================#
            S = prediction.shape[2]
            # ===========================================#
            #   构建网络应有的预测结果y_true
            #   shape: (B, 3, S, S, 5+80)
            # ===========================================#
            y_true = self.build_target(i, prediction, targets)
            # ===========================================#
            #   正样本mask
            # ===========================================#
            obj_mask = y_true[..., 4] == 1
            # if obj_mask.sum().item() > 0:
            #     boxes, scores, labels = [], [], []
            #     output = self._detect(i, y_true).squeeze()
            #     x1 = output[..., :4][:, 0] - output[..., :4][:, 2] / 2
            #     y1 = output[..., :4][:, 1] - output[..., :4][:, 3] / 2
            #     x2 = output[..., :4][:, 0] + output[..., :4][:, 2] / 2
            #     y2 = output[..., :4][:, 1] + output[..., :4][:, 3] / 2
            #     bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            #     _scores = (output[..., 4].unsqueeze(-1) * output[..., 5:])
            #     for cls_id in range(20):
            #         cls_scores = _scores[..., cls_id]
            #         keep = cls_scores > 0.45
            #         if keep.sum() == 0:
            #             continue
            #         cls_boxes = bboxes[keep]
            #         cls_scores = cls_scores[keep]
            #         cls_labels = torch.full((cls_scores.shape[0],), cls_id).long()

            #         boxes.extend(cls_boxes)
            #         scores.append(cls_scores)
            #         labels.append(cls_labels)
            #     image = item[0][0]
            #     from utils.boxTool import draw_box
            #     from detect import transport
            #     import cv2 as cv
            #     image = transport(image)[0]
            #     draw_box(image, boxes, scores, labels)
            #     cv.imshow("image", image)
            #     cv.waitKey(0)
            #     cv.destroyAllWindows()
            # x = prediction[obj_mask][:, 0].sigmoid() * S

            # y = prediction[obj_mask][:, 1].sigmoid() * S

            # best_a = obj_mask.nonzero()[:, 1]

            # w = torch.exp(prediction[obj_mask][:, 2]) * anchors[best_a][:, 0] / stride

            # h = torch.exp(prediction[obj_mask][:, 3]) * anchors[best_a][:, 1] / stride
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
                giou = self.compute_giou(i, S, prediction, y_true , obj_mask)
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

    def _fill_target(self, y_true, b, k, x, y, c, batch_target, anchors, index):
        """辅助函数：填充目标值"""
        y_true[b, k, x, y, 0] = batch_target[index, 0] - x.float()
        y_true[b, k, x, y, 1] = batch_target[index, 1] - y.float()
        y_true[b, k, x, y, 2] = torch.log(batch_target[index, 2] / anchors[k][0])
        y_true[b, k, x, y, 3] = torch.log(batch_target[index, 3] / anchors[k][1])
        y_true[b, k, x, y, 4] = 1
        y_true[b, k, x, y, 5 + c] = 1
        return y_true

    def build_target(self, i, prediction, targets):
        B = prediction.shape[0]
        S = prediction.shape[2]
        y_true = torch.zeros_like(prediction)

        for b in range(B):
            if targets[b].shape[0] == 0:  # 处理无目标的情况
                continue

            # 预处理目标框
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, 0:4] = targets[b][:, 0:4] * S
            batch_target[:, 4] = targets[b][:, 4]

            # 计算IOU矩阵
            gt_box = batch_target[:, 2:4]
            iou_matrix = self.compute_iou(gt_box, self.anchors[i])
            best_iou, best_na = torch.max(iou_matrix, dim=1)
            if best_iou.max() < 0.3:
                continue

            # 处理每个目标框
            for index, n_a in enumerate(best_na.tolist()):
                # 获取坐标和类别
                x = torch.clamp(batch_target[index, 0].long(), 0, S - 1)
                y = torch.clamp(batch_target[index, 1].long(), 0, S - 1)
                c = batch_target[index, 4].long()

                # 处理最佳匹配的anchor
                k = n_a
                y_true = self._fill_target(y_true, b, k, x, y, c, batch_target, self.anchors[i], index)

                # 处理其他匹配的anchor
                additional_anchors = (iou_matrix[index] > 0.5).nonzero().squeeze(1)
                for a in additional_anchors.tolist():
                    if a != n_a:
                        k = a
                        y_true = self._fill_target(y_true, b, k, x, y, c, batch_target, self.anchors[i], index)

        return y_true

    def compute_iou(self, gt_box, anchors):
        gt_wh = gt_box.unsqueeze(1)  # [N, 1, 2]
        anchors_wh = anchors.unsqueeze(0)  # [1, A, 2]

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
        return iou

    def compute_giou(self, i, S, prediction, y_true , obj_mask):
        b, a, gj, gi = obj_mask.nonzero(as_tuple=True)
        anchors = self.anchors[i][a]

        p_x = prediction[obj_mask][:, 0].sigmoid() * S
        p_y = prediction[obj_mask][:, 1].sigmoid() * S
        p_w = prediction[obj_mask][:, 2].exp() * anchors[:, 0]
        p_h = prediction[obj_mask][:, 3].exp() * anchors[:, 1]

        t_x = y_true[obj_mask][:, 0] * S
        t_y = y_true[obj_mask][:, 1] * S
        t_w = y_true[obj_mask][:, 2].exp() * anchors[:, 0]
        t_h = y_true[obj_mask][:, 3].exp() * anchors[:, 1]

        # xywh -> xyxy
        p_x1 = p_x - p_w / 2
        p_y1 = p_y - p_h / 2
        p_x2 = p_x + p_w / 2
        p_y2 = p_y + p_h / 2

        t_x1 = t_x - t_w / 2
        t_y1 = t_y - t_h / 2
        t_x2 = t_x + t_w / 2
        t_y2 = t_y + t_h / 2

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

        # 最小包围框
        xc1 = torch.min(p_x1, t_x1)
        yc1 = torch.min(p_y1, t_y1)
        xc2 = torch.max(p_x2, t_x2)
        yc2 = torch.max(p_y2, t_y2)
        area_c = ((xc2 - xc1) * (yc2 - yc1)).clamp(min=1e-6) + 1e-16

        giou = iou - (area_c - union) / (area_c)

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

    # def compute_mseloss(self, x, y, w, h, y_true, obj_mask, S, anchors, stride):
    #     best_a = obj_mask.nonzero()[:, 1]
    #     x_loss = ((x - y_true[obj_mask][:, 0] * S) ** 2).sum()
    #     y_loss = ((y - y_true[obj_mask][:, 1] * S) ** 2).sum()
    #     w_loss = ((w - torch.exp(y_true[obj_mask][:, 2]) * anchors[best_a][:, 1] / stride) ** 2).sum()
    #     h_loss = ((h - torch.exp(y_true[obj_mask][:, 3]) * anchors[best_a][:, 0] / stride) ** 2).sum()
    #     return x_loss + y_loss + w_loss + h_loss
    
    # def _detect(self, i, y_true):
    #     bs, _, ny, nx, _ = y_true.shape

    #     grid, anchor_grid = self._make_grid(nx, ny, i)

    #     xy, wh, conf = y_true.split((2, 2, 20 + 1), 4)
    #     xy = (xy + grid) * self.stride[i]  # xy
    #     wh = torch.exp(wh) * anchor_grid  # wh
    #     y = torch.cat((xy, wh, conf), 4)

    #     return y.view(bs, 3 * nx * ny, 25)
    # def _make_grid(self, nx=20, ny=20, i=0):
    #     """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
    #     anchors.
    #     """
    #     d = self.anchors[i].device
    #     t = self.anchors[i].dtype
    #     shape = 1, 3, ny, nx, 2  # grid shape
    #     y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    #     yv, xv = torch.meshgrid(y, x, indexing="ij")  # torch>=0.7 compatibility
    #     grid = torch.stack((xv, yv), 2).expand(shape)  # add grid offset, i.e. y = 2.0 * x - 0.5
    #     anchor_grid = (self.anchors[i] * self.stride[i]).view((1, 3, 1, 1, 2)).expand(shape)
    #     return grid, anchor_grid