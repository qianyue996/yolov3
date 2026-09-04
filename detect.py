import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch
from loguru import logger

from utils.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from utils.inference import (
    camera_detect,
    image_detect,
    screen_detect,
    video_detect,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv3 目标检测工具（支持图片、视频文件、摄像头与屏幕截屏）"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="检测输入源：摄像头编号（如 0）、视频文件路径（如 video.mp4）、屏幕截屏（screen）、或图片路径（如 test.jpg），默认 0",
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
        "--show",
        action="store_true",
        help="处理视频文件时同步弹出 OpenCV 窗口实时预览（按 q 可提前结束）",
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
        help="输出路径（图片默认 outputs/result_<文件名>.png，视频默认 outputs/result_<文件名>.mp4）",
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

    # 设备选择与 CPU 线程配置
    if args.device:
        target_device = args.device.lower()
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_device.startswith("cuda") and torch.cuda.is_available():
        device_name = "cuda"
        # GPU 推理时将 CPU 辅助线程数限制为轻量数量，避免 OpenMP 线程争抢打满 CPU
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        logger.info(f"使用 GPU 进行推理加速: {torch.cuda.get_device_name(0)}")
    else:
        device_name = "cpu"
        num_threads = (
            args.threads if args.threads is not None else (os.cpu_count() or 4)
        )
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        logger.info(
            f"使用 CPU 进行多核心推理加速（线程数: {torch.get_num_threads()} 核心）"
        )

    source = args.source
    source_path = Path(source)

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
    elif source_path.suffix.lower() in VIDEO_EXTENSIONS or (
        source_path.exists()
        and source_path.is_file()
        and source_path.suffix.lower() not in IMAGE_EXTENSIONS
    ):
        if not source_path.exists():
            logger.error(f"错误: 视频文件不存在: {source}")
            sys.exit(1)
        video_detect(
            video_path=source,
            output_path=args.output,
            checkpoint=args.checkpoint,
            device_name=device_name,
            verbose=args.verbose,
            show=args.show,
        )
    elif source_path.suffix.lower() in IMAGE_EXTENSIONS or (
        source_path.exists() and source_path.is_file()
    ):
        if not source_path.exists():
            logger.error(f"错误: 图片文件不存在: {source}")
            sys.exit(1)
        image_detect(
            image_path=source,
            output_path=args.output,
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
            logger.error(
                f"无法识别的输入源: {source}（支持 摄像头编号、视频文件、图片文件或 screen）"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
