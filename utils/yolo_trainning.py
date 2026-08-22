import copy

import torch


def xyxy2xywh(
    targets: list[torch.Tensor], size_w: int, size_h: int
) -> list[torch.Tensor]:
    _targets = []
    for _target in targets:
        # 归一化转特征层大小
        target = copy.deepcopy(_target)
        target[:, [0, 2]] = target[:, [0, 2]] * size_w
        target[:, [1, 3]] = target[:, [1, 3]] * size_h
        x = ((target[:, 0] + target[:, 2]) / 2).unsqueeze(1)
        y = ((target[:, 1] + target[:, 3]) / 2).unsqueeze(1)
        w = (target[:, 2] - target[:, 0]).unsqueeze(1)
        h = (target[:, 3] - target[:, 1]).unsqueeze(1)
        c = target[:, 4].unsqueeze(1)
        target = torch.cat([x, y, w, h, c], dim=1)
        _targets.append(target)

    return _targets
