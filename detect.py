import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image, ImageDraw

from utils import non_max_suppression

device = "cuda" if torch.cuda.is_available() else "cpu"
img_w = 416
img_h = 416

class_names: list[str] = []
anchors: torch.Tensor | None = None
anchors_mask: list[list[int]] = []
_model = None


def _load_model(path: str = "1000_0.2988.pth") -> None:
    global _model, class_names, anchors, anchors_mask
    if _model is not None:
        return
    _model = torch.load(path, map_location=device, weights_only=False)
    class_names = _model.class_names
    anchors = torch.tensor(_model.anchors, device=device)
    anchors_mask = _model.anchors_mask
    _model.eval()


resize = transforms.Resize((img_w, img_h))
to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4711, 0.4475, 0.4080), std=(0.2378, 0.2329, 0.2361)
        ),
    ]
)


def transform(image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
    resized_image = resize(image)
    to_tensor_image = to_tensor(resized_image)
    return resized_image, to_tensor_image


def _get_model() -> torch.nn.Module:
    _load_model()
    return _model


def secend_stage(outputs: list[torch.Tensor]) -> torch.Tensor:
    _outputs = []
    for i, output in enumerate(outputs):
        _, _, size_w, size_h, _ = output.shape
        stride = img_w / size_w
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
    outputs = _get_model()(image)
    outputs = secend_stage(outputs)
    results = non_max_suppression(outputs, conf_thres=0.01, iou_thres=0.45)
    return results


def camera_detect() -> None:
    cap = cv.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            logger.error("无法获取帧！")
            break
        image = Image.fromarray(img).convert("RGB")
        resized_image, input_image = transform(image)
        results = detect(input_image.unsqueeze(0).to(device))
        image_handler = ImageDraw.ImageDraw(resized_image)
        for result in results:
            score = float(result[4])
            class_id = int(result[5])
            label_text = f"{class_names[class_id]} {score}"
            x_min, y_min, x_max, y_max = list(map(int, result[:4]))
            text_x = x_min
            text_y = y_min - 15
            image_handler.rectangle(((x_min, y_min), (x_max, y_max)), outline="red")
            image_handler.text((text_x, text_y), label_text, fill="green")

        image = np.array(resized_image)
        cv.namedWindow("Camera", cv.WINDOW_NORMAL)
        cv.imshow("Camera", image)
        if cv.waitKey(1) == ord("q"):
            break
    # 释放资源
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    _load_model()
    camera_detect()
