import time
import torch
import torchvision
from loguru import logger


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    prediction = prediction[0]

    device = prediction.device

    bbox = prediction[..., :4].sigmoid()
    conf = prediction[..., 4].sigmoid()
    class_probs = prediction[..., 5:].sigmoid()

    conf = conf.unsqueeze(1) * class_probs
    
    xy = bbox[:, :2]*2 - 9.5
    wh = (bbox[:, 2:4]*2)**2

    return 1
