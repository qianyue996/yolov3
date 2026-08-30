from pathlib import Path

import torch
from loguru import logger

from nets.yolov3 import YoloBody
from utils import non_max_suppression
from utils.config import (
    DEFAULT_ANCHORS,
    DEFAULT_ANCHORS_MASK,
    DEFAULT_CLASSES_PATH,
    IMG_W,
)
from utils.decode import decode_preds

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open(DEFAULT_CLASSES_PATH) as _f:
        import yaml

        class_names = yaml.safe_load(_f)
except Exception:
    class_names = []

anchors: torch.Tensor | None = None
anchors_mask: list[list[int]] = []
_model = None


def _load_model(
    path: str | None = None, device_name: str | None = None
) -> None:
    global _model, anchors, anchors_mask, device
    if device_name is not None:
        device = device_name
    if _model is not None:
        return

    target_path = path or "1000_0.2988.pth"
    if Path(target_path).exists():
        _model = torch.load(target_path, map_location=device, weights_only=False)
    elif path is None and Path("weights/best.pth").exists():
        _model = torch.load("weights/best.pth", map_location=device, weights_only=False)
    else:
        logger.warning(
            f"未找到权重文件 ({target_path})，使用随机初始化的 YoloBody 模型"
        )
        _model = YoloBody(
            DEFAULT_ANCHORS,
            DEFAULT_ANCHORS_MASK,
            class_names if class_names else [f"cls_{i}" for i in range(80)],
        ).to(device)

    anchors = torch.tensor(_model.anchors, device=device)
    anchors_mask = _model.anchors_mask
    _model.eval()


def _get_model() -> torch.nn.Module:
    _load_model()
    if _model is None:
        raise RuntimeError("模型加载失败")
    return _model


def _get_device() -> torch.device:
    if _model is not None:
        return next(_model.parameters()).device
    return torch.device(device)


@torch.inference_mode()
def secend_stage(
    outputs: list[torch.Tensor], device: torch.device | None = None
) -> torch.Tensor:
    if device is None:
        device = _get_device()
    if anchors is None:
        raise RuntimeError("请先调用 _load_model() 加载模型")
    _outputs = []
    for i, output in enumerate(outputs):
        _, _, feat_h, feat_w, _ = output.shape
        stride = IMG_W / feat_h
        c = output.sigmoid()[..., 4:]

        scaled_anchors_l = anchors[anchors_mask[i]] / stride
        cx, cy, w, h, _, _ = decode_preds(
            output, scaled_anchors_l, feat_h, feat_w, device
        )

        # 转换为像素空间: cx, cy 已含 grid offset, 乘 stride 即可
        x = cx.unsqueeze(-1) * stride
        y = cy.unsqueeze(-1) * stride
        w = w.unsqueeze(-1) * stride
        h = h.unsqueeze(-1) * stride

        output = torch.cat([x, y, w, h, c], dim=-1).view(
            1, len(scaled_anchors_l) * feat_h * feat_w, len(class_names) + 5
        )
        _outputs.append(output)

    return torch.cat(_outputs, dim=1).squeeze()


@torch.inference_mode()
def detect(image: torch.Tensor) -> torch.Tensor:
    model = _get_model()
    out_device = next(model.parameters()).device
    outputs = model(image.to(out_device))
    outputs = secend_stage(outputs, device=out_device)
    results = non_max_suppression(outputs, conf_thres=0.45, iou_thres=0.45)
    return results
