import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def load_checkpoint(device, check_path, original_model):

    init_weight_dict = torch.load(check_path, map_location=device)['model']

    original_model_dict = original_model.model.state_dict()

    init_weight_backbone_dict = {k: v for k, v in init_weight_dict.items() if k.startswith("backbone.")}

    for index, item in enumerate(init_weight_backbone_dict.items()):

        key, init_weight_param = item
        # 去掉原模型中的 "backbone." 前缀
        original_key = key.replace("backbone.", "")
        
        if original_key in original_model_dict and original_model_dict[original_key].shape == init_weight_param.shape:
            # 如果形状一致，拷贝参数
            original_model_dict[original_key].data.copy_(init_weight_param.data)
            logging.info(f"成功复制：{original_key}")
        else:
            logging.warning(f"跳过复制：{original_key}（形状不匹配或没有找到相应层）")

    # 5. 更新原始模型的参数
    original_model_dict.update(original_model_dict)

    original_model.model.load_state_dict(original_model_dict)

    logging.info("load checkpoint successfully！")