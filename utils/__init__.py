import yaml
import torch
import numpy as np
import random
from .loss import YOLOLOSS
from .nms import non_max_suppression


def load_classes(conf_path: str):
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


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


__all__ = [
    "load_classes",
    "YOLOLOSS",
    "set_seed",
    "worker_init_fn",
    "non_max_suppression",
]
