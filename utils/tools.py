import numpy as np
import cv2 as cv
import torch
import os
import shutil
def xywh2xyx2(box: list):
    try:
        x,y,w,h = box
        x1,y1 = int(x-w/2),int(y-h/2)
        x2,y2 = int(x+w/2),int(y+h/2)
    except Exception as e:
        print(f'错误：{e}')
    return [x1,y1,x2,y2]

def xyxy2xywh(box: list):
    try:
        x1,y1,x2,y2 = box
        x,y = int((x1+x2)/2),int((y1+y2)/2)
        w,h = int(x2-x1),int(y2-y1)
    except Exception as e:
        print('检查是否x2<x1 / y2<y1')
    return [x,y,w,h]
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持CV2读入的BRG转RGB
#---------------------------------------------------------#
def cvtColor(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

class Dynamic_lr():
    def __init__(self):
        super(Dynamic_lr, self).__init__()
        self.losses = []
    def __call__(self, optimizer, lr, loss):
        self.losses.append(loss)
        new_lr = lr
        try:
            if len(self.losses) == 2:
                if self.losses[-1] > self.losses[-2]:
                    new_lr = lr - lr * 0.1
                else:
                    new_lr = lr + lr * 0.1
                new_lr = max(min(new_lr, 1e-2), 1e-6)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.losses = []
            return new_lr
        except Exception as e:
            pass

def buildBox(i, S, stride, pred, anchors, anchors_mask, score_thresh=0.4):
    grid_x, grid_y = torch.meshgrid(
        torch.arange(S, dtype=pred.dtype, device=pred.device),
        torch.arange(S, dtype=pred.dtype, device=pred.device),
        indexing='ij'
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand_as(pred[..., 0:2])
    pred[..., 0:2] = (pred[..., 0:2].sigmoid() + grid_xy) * stride

    pred[..., 2] = torch.exp(pred[..., 2]) * anchors[anchors_mask[i]][:, 0].unsqueeze(-1).unsqueeze(-1)
    pred[..., 3] = torch.exp(pred[..., 3]) * anchors[anchors_mask[i]][:, 1].unsqueeze(-1).unsqueeze(-1)

    boxes = torch.stack([pred[..., 0] - pred[..., 2] / 2,
                        pred[..., 1] - pred[..., 3] / 2,
                        pred[..., 0] + pred[..., 2] / 2,
                        pred[..., 1] + pred[..., 3] / 2
                        ], dim=-1)
    boxes = torch.clamp(boxes, min=0, max=416)
    #===========================================#
    #   模型预测为正样本的输出
    #===========================================#
    pos_conf = pred[..., 4].sigmoid()
    #===========================================#
    #   模型预测的类别输出
    #===========================================#
    cls_conf = pred[..., 5:].sigmoid()
    #===========================================#
    #   模型预测所有位置的所有类别分数
    #   shape: 3, 13, 13, 80
    #===========================================#
    scores = pos_conf.unsqueeze(-1) * cls_conf
    # 选出所有分数大于阈值的 box + 类别
    for cls_id in range(80):
        cls_scores = scores[..., cls_id]
        keep = cls_scores > score_thresh
        if keep.sum() == 0:
            continue
        cls_boxes = boxes[keep]
        cls_scores = cls_scores[keep]
        cls_labels = torch.full((cls_scores.shape[0],), cls_id, dtype=torch.int64, device=CONF.device)
    pass

def nms(pred, target, iou_threshold=0.45):
    if len(boxes) != len(scores):
        print('boxes and scores length is not equal!')
        return
    
def compute_iou(box_a, box_b):
    '''
    box_a = [x1, y1, x2, y2] shape: (N, 4)
    box_b = [x1, y1, x2, y2] shape: (N, 4)
    '''
    min_xy = torch.min(box_a[:, :2], box_b[:, :2])
    max_xy = torch.max(box_a[:, 2:], box_b[:, 2:])

    inter = max_xy - min_xy
    pass

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def clean_folder(folder_path='runs', keep_last=5):
    if not os.path.exists(folder_path):
        print(f"{folder_path} 不存在")
        return

    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path)]
    subfolders.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # 保留最新 keep_last 个，其余都删
    to_delete = subfolders[keep_last:]

    for folder in to_delete:
        shutil.rmtree(folder)

class DynamicLr():
    '''
    主要传入optimizer和step_size
    说明：
        - optimizer 优化器
        - step_size 观察次数，每多少步观察一次 loss 均值
    实现：
        - 每隔 step_size 步，观察 loss 均值
        - 如果 loss 均值没有明显下降，则将学习率乘以 decay_factor
        - 如果 loss 均值下降明显，则将学习率乘以 boost_factor
    '''
    def __init__(self, optimizer, step_size=5, max_lr=0.01, decay_factor=0.96, boost_factor=1.05):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.step_size = step_size  # 每多少步观察一次 loss 均值
        self.decay_factor = decay_factor  # 降速因子
        self.boost_factor = boost_factor  # 升速因子

        self.run_step = 1
        self.loss_history = []
        self.last_avg_loss = None

    def step(self, current_loss):
        #===========================================#
        #   Loss均值计数
        #===========================================#
        self.loss_history.append(current_loss)
        avg_loss = np.array(self.loss_history).mean()
        #===========================================#
        #   判断step是否大于step_size，选择操作
        #===========================================#
        if self.run_step > self.step_size:
            delta = self.last_avg_loss - avg_loss
            # loss 没明显下降
            if delta < 1e-4:
                self._adjust_lr(decay=True)
            # loss 下降明显
            elif delta > 1e-2:
                self._adjust_lr(boost=True)
        #===========================================#
        #   更新last_avg_loss和step
        #===========================================#
        self.last_avg_loss = avg_loss
        self.run_step += 1
        
    def _adjust_lr(self, decay=False, boost=False):
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            if decay:
                new_lr = old_lr * self.decay_factor
            elif boost:
                new_lr = min(old_lr * self.boost_factor, self.max_lr)
            else:
                new_lr = old_lr
            group['lr'] = new_lr
        self.run_step = 1
