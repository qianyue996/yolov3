import torch

from utils import non_max_suppression
from utils.config import IMG_W

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open("data/coco_names.yaml") as _f:
        import yaml
        class_names = yaml.safe_load(_f)
except Exception:
    class_names = []

anchors: torch.Tensor | None = None
anchors_mask: list[list[int]] = []
_model = None


def _load_model(path: str = "1000_0.2988.pth") -> None:
    global _model, anchors, anchors_mask
    if _model is not None:
        return
    _model = torch.load(path, map_location=device, weights_only=False)
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


def secend_stage(
    outputs: list[torch.Tensor], device: torch.device | None = None
) -> torch.Tensor:
    if device is None:
        device = _get_device()
    if anchors is None:
        raise RuntimeError("请先调用 _load_model() 加载模型")
    _outputs = []
    for i, output in enumerate(outputs):
        _, _, size_w, size_h, _ = output.shape
        stride = IMG_W / size_w
        x = output.sigmoid()[..., 0] * 2 - 0.5
        y = output.sigmoid()[..., 1] * 2 - 0.5
        w = (output.sigmoid()[..., 2] * 2) ** 2
        h = (output.sigmoid()[..., 3] * 2) ** 2
        c = output.sigmoid()[..., 4:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(size_h, device=device),
            torch.arange(size_w, device=device),
            indexing="ij",
        )
        scaled_anchors_l = anchors[anchors_mask[i]]
        anchor_w = scaled_anchors_l[:, 0].view(1, -1, 1, 1).expand_as(w)
        anchor_h = scaled_anchors_l[:, 1].view(1, -1, 1, 1).expand_as(h)
        x = torch.unsqueeze(x + grid_x, -1) * stride
        y = torch.unsqueeze(y + grid_y, -1) * stride
        w = torch.unsqueeze(w * anchor_w, -1) * stride
        h = torch.unsqueeze(h * anchor_h, -1) * stride
        output = torch.cat([x, y, w, h, c], dim=-1).view(
            1, len(scaled_anchors_l) * size_w * size_h, len(class_names) + 5
        )
        _outputs.append(output)

    return torch.cat(_outputs, dim=1).squeeze()


def detect(image: torch.Tensor) -> torch.Tensor:
    model = _get_model()
    out_device = next(model.parameters()).device
    outputs = model(image.to(out_device))
    outputs = secend_stage(outputs, device=out_device)
    results = non_max_suppression(outputs, conf_thres=0.01, iou_thres=0.45)
    return results
