import torch.nn as nn
import math
import torch.nn.functional as F

def initialize_weights(model):
    """Initializes weights for Conv2D, BatchNorm2d, and activation layers (Hardswish, LeakyReLU, ReLU, ReLU6, SiLU) in a
    model.
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scales and optionally pads an image tensor to a specified ratio, maintaining its aspect ratio constrained by
    `gs`.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean