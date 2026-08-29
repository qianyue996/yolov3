import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from utils import _get_model, _load_model, class_names, detect
from utils.config import IMG_H, IMG_W
from utils.transforms import image_transform

try:
    _font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    _font = ImageFont.load_default()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


def draw_detections(
    pil_image: Image.Image,
    results: torch.Tensor,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Image.Image:
    """在图片上绘制检测框、类别名与置信度。

    Args:
        pil_image: 待绘制的 PIL 图像
        results: NMS 输出的检测结果张量 (N, 6)，每行 [x1, y1, x2, y2, score, class_id]
        scale_x: x 方向从 416 映射回原图的缩放比例
        scale_y: y 方向从 416 映射回原图的缩放比例

    Returns:
        绘制完成的 PIL 图像
    """
    img_w, img_h = pil_image.size
    drawer = ImageDraw.ImageDraw(pil_image)

    for result in results:
        score = float(result[4])
        class_id = int(result[5])
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label_text = f"{class_name} {score:.2f}"

        x_min, y_min, x_max, y_max = map(float, result[:4])
        x_min = max(0, min(int(x_min * scale_x), img_w))
        y_min = max(0, min(int(y_min * scale_y), img_h))
        x_max = max(0, min(int(x_max * scale_x), img_w))
        y_max = max(0, min(int(y_max * scale_y), img_h))

        drawer.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)

        text_x = x_min
        text_y = y_min - 20 if y_min >= 20 else y_min + 2
        text_bbox = drawer.textbbox((text_x, text_y), label_text, font=_font)
        drawer.rectangle(text_bbox, fill="red")
        drawer.text((text_x, text_y), label_text, fill="white", font=_font)

    return pil_image


def image_detect(
    image_path: str,
    output_path: str | None = None,
    checkpoint: str | None = None,
) -> None:
    """单张图片目标检测并保存结果。"""
    if checkpoint:
        _load_model(checkpoint)
    else:
        _load_model()

    out_device = next(_get_model().parameters()).device
    pil_image = Image.open(image_path).convert("RGB")
    original_size = pil_image.size

    _, input_tensor = image_transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(out_device)

    results = detect(input_tensor)

    scale_x = original_size[0] / IMG_W
    scale_y = original_size[1] / IMG_H
    draw_detections(pil_image, results, scale_x=scale_x, scale_y=scale_y)

    out = (
        Path(output_path)
        if output_path
        else output_dir / f"result_{Path(image_path).stem}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out)
    logger.info(f"图片检测完成，检测到 {len(results)} 个目标，结果已保存至: {out}")


def camera_detect(
    camera_id: int = 0,
    checkpoint: str | None = None,
) -> None:
    """摄像头实时视频流目标检测。"""
    if checkpoint:
        _load_model(checkpoint)
    else:
        _load_model()

    out_device = next(_get_model().parameters()).device
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 (ID: {camera_id})！")
        return

    logger.info(f"摄像头检测已启动 (ID: {camera_id})，按 'q' 键退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("无法获取视频帧！")
            break

        pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        original_size = pil_image.size

        _, input_tensor = image_transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(out_device)

        results = detect(input_tensor)

        scale_x = original_size[0] / IMG_W
        scale_y = original_size[1] / IMG_H
        draw_detections(pil_image, results, scale_x=scale_x, scale_y=scale_y)

        annotated_frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        cv.namedWindow("YOLOv3 Detection", cv.WINDOW_NORMAL)
        cv.imshow("YOLOv3 Detection", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("摄像头检测已结束。")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv3 目标检测工具（自动支持图片与摄像头）")
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="检测输入源：摄像头索引（如 0）或图片文件路径（如 img/test.jpg），默认 0（摄像头）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="图片检测时的输出路径（默认 outputs/result_<原文件名>.png）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型权重文件路径（默认使用已有 checkpoint）",
    )
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        camera_detect(camera_id=int(source), checkpoint=args.checkpoint)
    elif Path(source).exists() or any(
        source.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ):
        if not Path(source).exists():
            logger.error(f"错误: 图片文件不存在: {source}")
            sys.exit(1)
        image_detect(image_path=source, output_path=args.output, checkpoint=args.checkpoint)
    else:
        try:
            cam_id = int(source)
            camera_detect(camera_id=cam_id, checkpoint=args.checkpoint)
        except ValueError:
            logger.error(f"无法识别的输入源: {source}（必须是摄像头编号或有效图片路径）")
            sys.exit(1)


if __name__ == "__main__":
    main()
