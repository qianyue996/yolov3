import random
import torchvision.ops as ops
import numpy as np
import torch
import yaml


def set_seed(seed=27):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    random.seed(seed)


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) != len(scores):
        print("boxes and scores length is not equal!")
        return
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        current = idxs[0]
        keep.append(current.item())

        if idxs.numel() == 1:
            break

        ious = compute_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]

    return keep


def compute_iou(box_a, box_b):
    """
    box_a = [x1, y1, x2, y2] shape: (4)
    box_b = [x1, y1, x2, y2] shape: (N, 4)
    """
    box_a_wh = box_a[2:] - box_a[:2]
    box_b_wh = box_b[:, 2:] - box_b[:, :2]

    area_a = box_a_wh.prod()
    area_b = box_b_wh.prod(dim=1)

    area_x1y1 = torch.max(box_a[:2], box_b[:, :2])
    area_x2y2 = torch.min(box_a[2:], box_b[:, 2:])

    area_wh = (area_x2y2 - area_x1y1).clamp(min=0)

    inter = area_wh[:, 0] * area_wh[:, 1]

    union = area_a + area_b - inter

    return inter / union


class DynamicLr:
    """
    主要传入optimizer和step_size
    说明：
        - optimizer 优化器
        - step_size 观察次数，每多少步观察一次 loss 均值
        - lr 如果非None，则使用此学习率初始化优化器的学习率；否则沿用优化器原有学习率
    实现：
        - 每隔 step_size 步，观察 loss 均值
        - 如果 loss 均值没有明显下降，则将学习率乘以 decay_factor
        - 如果 loss 均值下降明显，则将学习率乘以 boost_factor

    """

    def __init__(
        self,
        optimizer,
        step_size=5,
        init_lr=None,
        max_lr=0.01,
        min_lr=1e-4,
        decay_factor=0.5,
        boost_factor=2,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size  # 每多少步观察一次 loss 均值
        self.decay_factor = decay_factor  # 降速因子
        self.boost_factor = boost_factor  # 升速因子
        # 如果 init_lr 非 None，则设置优化器的学习率为 init_lr
        if init_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = init_lr

        self.run_step = 1
        self.loss_history = []

    def step(self, current_loss):
        # ===========================================#
        #   Loss均值计数
        # ===========================================#
        self.loss_history.append(current_loss)
        loss_arr = np.array(self.loss_history[: -self.run_step])
        if loss_arr.size == 0:
            last_avg_loss = 0
        else:
            last_avg_loss = loss_arr.mean()
        # ===========================================#
        #   判断step是否大于step_size，选择操作
        # ===========================================#
        if self.run_step > self.step_size:
            # ===========================================#
            #   实现：
            #       - avg_loss：全局loss减去最新step_size个loss均值
            #       - last_avg_loss：全局loss均值
            #       - 目的是看最新加入step_size个loss的loss均值
            #       - 如果loss没明显下降，则将学习率乘以 decay_factor
            #       - 如果loss下降明显，则将学习率乘以 boost_factor
            # ===========================================#
            avg_loss = np.array(self.loss_history).mean()
            delta = avg_loss - last_avg_loss
            # loss 没明显下降
            if delta < 1e-4:
                self._adjust_lr(decay=True)
            # loss 下降明显
            elif delta > 1e-2:
                self._adjust_lr(boost=True)
        # ===========================================#
        #   更新step
        # ===========================================#
        self.run_step += 1

    def _adjust_lr(self, decay=False, boost=False):
        for group in self.optimizer.param_groups:
            old_lr = group["lr"]
            if decay:
                new_lr = max(old_lr * self.decay_factor, self.min_lr)
            elif boost:
                new_lr = min(old_lr * self.boost_factor, self.max_lr)
            else:
                new_lr = old_lr
            group["lr"] = new_lr
        self.run_step = 1


def multi_class_nms(boxes, scores, labels, iou_thres=0.5):
    keep_boxes, keep_scores, keep_labels = [], [], []
    unique_labels = labels.unique()
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = ops.nms(cls_boxes, cls_scores, iou_thres)
        keep_boxes.append(cls_boxes[keep])
        keep_scores.append(cls_scores[keep])
        keep_labels.append(labels[cls_mask][keep])
    return (
        torch.cat(keep_boxes),
        torch.cat(keep_scores),
        torch.cat(keep_labels),
    )
