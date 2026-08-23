import random

import numpy as np
import torch
import yaml

from .dataloader import yolo_collate_fn
from .loss import YOLOLOSS
from .nms import non_max_suppression
from .postprocess import (
    _get_device,
    _get_model,
    _load_model,
    anchors,
    anchors_mask,
    class_names,
    detect,
    device,
    secend_stage,
)
from .transforms import IMG_H, IMG_W, transform


def load_classes(conf_path: str) -> list:
    with open(conf_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 27) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    random.seed(seed)


__all__ = [
    "load_classes",
    "set_seed",
    "worker_init_fn",
    "YOLOLOSS",
    "non_max_suppression",
    # transforms
    "IMG_W",
    "IMG_H",
    "transform",
    "yolo_collate_fn",
    # postprocess
    "device",
    "class_names",
    "anchors",
    "anchors_mask",
    "_load_model",
    "_get_model",
    "_get_device",
    "secend_stage",
    "detect",
]
