import argparse
import contextlib
import os
import sys
import time
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


def _format_detection_log(results: torch.Tensor, elapsed_s: float) -> str:
    """格式化单帧检测性能与目标统计日志。

    Args:
        results: NMS 检测结果张量 (N, 6)
        elapsed_s: 单帧端到端耗时（秒）

    Returns:
        包含耗时、FPS 及目标数量和分类的统计字符串
    """
    fps = 1.0 / elapsed_s if elapsed_s > 0 else float("inf")
    num_objects = len(results)
    if num_objects == 0:
        return f"耗时: {elapsed_s * 1000:.1f}ms ({fps:.1f} FPS) | 检测到 0 个目标"

    class_counts: dict[str, int] = {}
    for res in results:
        cid = int(res[5])
        cname = class_names[cid] if cid < len(class_names) else str(cid)
        class_counts[cname] = class_counts.get(cname, 0) + 1

    details = ", ".join(f"{name}: {cnt}" for name, cnt in class_counts.items())
    return f"耗时: {elapsed_s * 1000:.1f}ms ({fps:.1f} FPS) | 检测到 {num_objects} 个目标 [{details}]"


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
    device_name: str | None = None,
    verbose: bool = False,
) -> None:
    """单张图片目标检测并保存结果。"""
    if checkpoint:
        _load_model(checkpoint, device_name=device_name)
    else:
        _load_model(device_name=device_name)

    out_device = next(_get_model().parameters()).device
    t0 = time.perf_counter()
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
    elapsed_s = time.perf_counter() - t0

    if verbose:
        logger.info(f"[图片检测] {_format_detection_log(results, elapsed_s)}")
    else:
        logger.info(f"图片检测完成，检测到 {len(results)} 个目标，结果已保存至: {out}")


def screen_detect(
    monitor_id: int = 1,
    checkpoint: str | None = None,
    device_name: str | None = None,
    verbose: bool = False,
) -> None:
    """持续截图屏幕中间 416x416 正方形区域进行实时目标检测并显示。

    Args:
        monitor_id: 目标显示器编号（1 为主显示器，0 为全屏跨屏）
        checkpoint: 模型权重文件路径
        device_name: 运算设备 ('cuda' 或 'cpu')
        verbose: 是否持续打印速度与目标统计日志
    """
    if checkpoint:
        _load_model(checkpoint, device_name=device_name)
    else:
        _load_model(device_name=device_name)

    out_device = next(_get_model().parameters()).device

    import mss

    try:
        sct = mss.mss()
    except Exception as e:
        logger.error(f"无法初始化屏幕截图 (mss): {e}")
        return

    monitors = sct.monitors
    if monitor_id >= len(monitors):
        logger.warning(
            f"指定的显示器编号 {monitor_id} 超出范围 (共 {len(monitors)} 个)，自动选择主显示器"
        )
        monitor_id = 1 if len(monitors) > 1 else 0

    mon = monitors[monitor_id]
    left = mon["left"] + max(0, (mon["width"] - IMG_W) // 2)
    top = mon["top"] + max(0, (mon["height"] - IMG_H) // 2)
    bbox = {"left": int(left), "top": int(top), "width": IMG_W, "height": IMG_H}

    logger.info(
        f"屏幕截屏检测已启动 (显示器 {monitor_id}, 区域: left={left}, top={top}, size={IMG_W}x{IMG_H})，按 'q' 键退出..."
    )

    cv.namedWindow("YOLOv3 Screen Detection", cv.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            sct_img = sct.grab(bbox)
            pil_image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            _, input_tensor = image_transform(pil_image)
            input_tensor = input_tensor.unsqueeze(0).to(out_device)

            results = detect(input_tensor)
            draw_detections(pil_image, results, scale_x=1.0, scale_y=1.0)

            annotated_frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
            cv.imshow("YOLOv3 Screen Detection", annotated_frame)

            elapsed_s = time.perf_counter() - t0
            if verbose:
                logger.info(f"[屏幕检测] {_format_detection_log(results, elapsed_s)}")

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        logger.error(f"屏幕截图检测运行异常: {e}")
    finally:
        sct.close()
        cv.destroyAllWindows()
        logger.info("屏幕截屏检测已结束。")


def camera_detect(
    camera_id: int = 0,
    checkpoint: str | None = None,
    device_name: str | None = None,
    verbose: bool = False,
) -> None:
    """摄像头实时视频流目标检测。

    Args:
        camera_id: 摄像头设备 ID
        checkpoint: 模型权重文件路径
        device_name: 运算设备 ('cuda' 或 'cpu')
        verbose: 是否持续打印速度与目标统计日志
    """
    if checkpoint:
        _load_model(checkpoint, device_name=device_name)
    else:
        _load_model(device_name=device_name)

    out_device = next(_get_model().parameters()).device
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 (ID: {camera_id})！")
        return

    logger.info(f"摄像头检测已启动 (ID: {camera_id})，按 'q' 键退出...")

    while True:
        t0 = time.perf_counter()
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

        elapsed_s = time.perf_counter() - t0
        if verbose:
            logger.info(f"[摄像头检测] {_format_detection_log(results, elapsed_s)}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("摄像头检测已结束。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv3 目标检测工具（支持图片、摄像头与屏幕截屏）"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="检测输入源：摄像头索引（如 0）、屏幕截屏（screen）、或图片路径（如 img/test.jpg），默认 0",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="直接开启屏幕截屏检测（持续截取屏幕中心 416x416 正方形区域）",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="屏幕截屏时使用的显示器编号（默认 1 为主显示器）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        "--log",
        action="store_true",
        dest="verbose",
        help="持续打印日志：输出处理耗时 (ms)、FPS 及检测到的目标类别与数量",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备（默认自动检测：优先使用 GPU，不可用时回退到 CPU 多核）",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU 推理线程数（仅在 CPU 推理时生效，默认自动使用全部物理核心数）",
    )
    args = parser.parse_args()

    # 设备选择与 CPU 多核配置逻辑
    if args.device:
        target_device = args.device.lower()
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_device.startswith("cuda") and torch.cuda.is_available():
        device_name = "cuda"
        logger.info(f"使用 GPU 进行推理加速: {torch.cuda.get_device_name(0)}")
    else:
        device_name = "cpu"
        num_threads = args.threads if args.threads is not None else (os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        logger.info(
            f"使用 CPU 进行多核心推理加速（线程数: {torch.get_num_threads()} 核心）"
        )

    source = args.source
    if args.screen or source.lower().startswith("screen"):
        monitor_id = args.monitor
        if ":" in source:
            with contextlib.suppress(ValueError):
                monitor_id = int(source.split(":", 1)[1])
        screen_detect(
            monitor_id=monitor_id,
            checkpoint=args.checkpoint,
            device_name=device_name,
            verbose=args.verbose,
        )
    elif source.isdigit():
        camera_detect(
            camera_id=int(source),
            checkpoint=args.checkpoint,
            device_name=device_name,
            verbose=args.verbose,
        )
    elif Path(source).exists() or any(
        source.lower().endswith(ext)
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ):
        if not Path(source).exists():
            logger.error(f"错误: 图片文件不存在: {source}")
            sys.exit(1)
        image_detect(
            image_path=source,
            output_path=args.output,
            checkpoint=args.checkpoint,
            device_name=device_name,
            verbose=args.verbose,
        )
    else:
        try:
            cam_id = int(source)
            camera_detect(
                camera_id=cam_id,
                checkpoint=args.checkpoint,
                device_name=device_name,
                verbose=args.verbose,
            )
        except ValueError:
            logger.error(f"无法识别的输入源: {source}（支持 0/摄像头编号、screen、或图片路径）")
            sys.exit(1)


if __name__ == "__main__":
    main()
