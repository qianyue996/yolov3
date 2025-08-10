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

    bbox = prediction[..., :4]
    conf = prediction[..., 4]
    class_probs = prediction[..., 5:]

    

    return 1
