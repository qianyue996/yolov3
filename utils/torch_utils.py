import torch.nn as nn
import math
import torch
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

def load_checkpoint(device, check_path, original_model):
    my_model_dict = torch.load(check_path, map_location=device)['model']
    original_model_dict = original_model.model.state_dict()

    my_backbone_dict = {k: v for k, v in my_model_dict.items() if k.startswith("backbone.")}

    for key, my_param in my_backbone_dict.items():
        # 去掉原模型中的 "backbone." 前缀
        original_key = key.replace("backbone.", "")
        
        if original_key in original_model_dict and original_model_dict[original_key].shape == my_param.shape:
            # 如果形状一致，拷贝参数
            print(f"复制：{original_key}")
            original_model_dict[original_key].data.copy_(my_param.data)
        else:
            print(f"跳过：{original_key}（形状不匹配或没有找到相应层）")

    # 5. 更新原始模型的参数
    original_model_dict.update(original_model_dict)
    original_model.model.load_state_dict(original_model_dict)
    print("load checkpoint successfully！")