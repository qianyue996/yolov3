import shutil
import subprocess
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from utils.config import (
    IMG_H,
    IMG_W,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from utils.postprocess import _get_model, _load_model, class_names, detect

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 预先分配好 GPU 归一化常量，避免在推理循环中反复创建
_NORM_MEAN_TENSOR: torch.Tensor | None = None
_NORM_STD_TENSOR: torch.Tensor | None = None


def _get_norm_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    global _NORM_MEAN_TENSOR, _NORM_STD_TENSOR
    if (
        _NORM_MEAN_TENSOR is None
        or _NORM_STD_TENSOR is None
        or _NORM_MEAN_TENSOR.device != device
    ):
        _NORM_MEAN_TENSOR = torch.tensor(
            NORMALIZE_MEAN, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        _NORM_STD_TENSOR = torch.tensor(
            NORMALIZE_STD, device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
    return _NORM_MEAN_TENSOR, _NORM_STD_TENSOR


def image_to_tensor_gpu(
    frame_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    device: torch.device,
) -> torch.Tensor:
    """高性能图像预处理：直接将图像送入目标设备（GPU）进行 float 转换与归一化。

    消除了 CPU PIL/Transforms 重复拷贝与单核计算瓶颈。

    Args:
        frame_bgr: BGR 格式的 numpy 图像 (H, W, 3)
        target_w: 模型输入宽度（如 416）
        target_h: 模型输入高度（如 416）
        device: 计算设备（CUDA 或 CPU）

    Returns:
        归一化后的模型输入张量 (1, 3, target_h, target_w)
    """
    if frame_bgr.shape[1] != target_w or frame_bgr.shape[0] != target_h:
        resized_bgr = cv.resize(
            frame_bgr, (target_w, target_h), interpolation=cv.INTER_LINEAR
        )
    else:
        resized_bgr = frame_bgr

    # OpenCV 快速色彩转换 (BGR -> RGB)
    rgb = cv.cvtColor(resized_bgr, cv.COLOR_BGR2RGB)

    # 异步推送到 GPU，并在 GPU 上执行向量化浮点除法与均值方差归一化
    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device, non_blocking=True)
        .float()
    )
    tensor.div_(255.0)

    mean_t, std_t = _get_norm_tensors(device)
    return (tensor - mean_t) / std_t


def _format_detection_log(results: torch.Tensor | np.ndarray, elapsed_s: float) -> str:
    """格式化单帧检测性能与目标统计日志。"""
    fps = 1.0 / elapsed_s if elapsed_s > 0 else float("inf")
    num_objects = len(results)
    if num_objects == 0:
        return f"耗时: {elapsed_s * 1000:.1f}ms ({fps:.1f} FPS) | 检测到 0 个目标"

    if isinstance(results, torch.Tensor):
        results_np = results.detach().cpu().numpy()
    else:
        results_np = results

    class_counts: dict[str, int] = {}
    for res in results_np:
        cid = int(res[5])
        cname = class_names[cid] if cid < len(class_names) else str(cid)
        class_counts[cname] = class_counts.get(cname, 0) + 1

    details = ", ".join(f"{name}: {cnt}" for name, cnt in class_counts.items())
    return f"耗时: {elapsed_s * 1000:.1f}ms ({fps:.1f} FPS) | 检测到 {num_objects} 个目标 [{details}]"


def draw_detections_cv(
    frame: np.ndarray,
    results: torch.Tensor | np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """直接使用 OpenCV 在 BGR 图像上绘制检测框与标签（单批次转移，零 CPU 阻塞）。

    Args:
        frame: BGR 图像 (H, W, 3)
        results: 检测结果 (N, 6)，[x1, y1, x2, y2, score, class_id]
        scale_x: x 坐标缩放映射比例
        scale_y: y 坐标缩放映射比例

    Returns:
        绘制完成的 BGR 图像
    """
    if isinstance(results, torch.Tensor):
        results_np = results.detach().cpu().numpy()
    else:
        results_np = results

    if len(results_np) == 0:
        return frame

    img_h, img_w = frame.shape[:2]

    for row in results_np:
        x_min, y_min, x_max, y_max, score, class_id = row
        class_id = int(class_id)
        class_name = (
            class_names[class_id] if class_id < len(class_names) else str(class_id)
        )
        label = f"{class_name} {score:.2f}"

        x1 = max(0, min(int(x_min * scale_x), img_w - 1))
        y1 = max(0, min(int(y_min * scale_y), img_h - 1))
        x2 = max(0, min(int(x_max * scale_x), img_w - 1))
        y2 = max(0, min(int(y_max * scale_y), img_h - 1))

        # 绘制红色检测框
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 绘制文本背景框与标签
        (tw, th), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 4)
        cv.rectangle(
            frame,
            (x1, ty - th - 4),
            (x1 + tw + 4, ty + baseline - 2),
            (0, 0, 255),
            -1,
        )
        cv.putText(
            frame,
            label,
            (x1 + 2, ty - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

    return frame


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

    frame_bgr = cv.imread(image_path)
    if frame_bgr is None:
        from PIL import Image

        pil_img = Image.open(image_path).convert("RGB")
        frame_bgr = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

    orig_h, orig_w = frame_bgr.shape[:2]
    input_tensor = image_to_tensor_gpu(frame_bgr, IMG_W, IMG_H, out_device)

    results = detect(input_tensor)

    scale_x = orig_w / IMG_W
    scale_y = orig_h / IMG_H
    draw_detections_cv(frame_bgr, results, scale_x=scale_x, scale_y=scale_y)

    out = (
        Path(output_path)
        if output_path
        else output_dir / f"result_{Path(image_path).stem}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out), frame_bgr)
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
            # 直接从内存 buffer 转换为 numpy BGR 图像（零 PIL 转换）
            frame_bgra = np.asarray(sct_img)
            frame_bgr = frame_bgra[:, :, :3].copy()

            input_tensor = image_to_tensor_gpu(frame_bgr, IMG_W, IMG_H, out_device)
            results = detect(input_tensor)

            draw_detections_cv(frame_bgr, results, scale_x=1.0, scale_y=1.0)
            cv.imshow("YOLOv3 Screen Detection", frame_bgr)

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
        ret, frame_bgr = cap.read()
        if not ret:
            logger.error("无法获取视频帧！")
            break

        orig_h, orig_w = frame_bgr.shape[:2]
        input_tensor = image_to_tensor_gpu(frame_bgr, IMG_W, IMG_H, out_device)

        results = detect(input_tensor)

        scale_x = orig_w / IMG_W
        scale_y = orig_h / IMG_H
        draw_detections_cv(frame_bgr, results, scale_x=scale_x, scale_y=scale_y)

        cv.namedWindow("YOLOv3 Detection", cv.WINDOW_NORMAL)
        cv.imshow("YOLOv3 Detection", frame_bgr)

        elapsed_s = time.perf_counter() - t0
        if verbose:
            logger.info(f"[摄像头检测] {_format_detection_log(results, elapsed_s)}")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("摄像头检测已结束。")


def _merge_audio(
    temp_video_path: Path, source_video_path: str, output_path: Path
) -> None:
    """将原视频的音频流与检测标注后的视频画面合流，保留原声音轨。"""
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        logger.warning("未检测到系统 ffmpeg 工具，视频将输出无声音版本。")
        if temp_video_path != output_path:
            temp_video_path.replace(output_path)
        return

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(temp_video_path),
        "-i",
        str(source_video_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
        if temp_video_path.exists():
            temp_video_path.unlink()
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"音频合流失败 ({e.stderr.decode().strip()})，保留无声音版本视频。"
        )
        if temp_video_path != output_path:
            temp_video_path.replace(output_path)


def video_detect(
    video_path: str,
    output_path: str | None = None,
    checkpoint: str | None = None,
    device_name: str | None = None,
    verbose: bool = False,
    show: bool = False,
) -> None:
    """对输入视频文件进行逐帧目标检测，并生成保存标注后的新视频（保留原始声音）。

    Args:
        video_path: 输入视频文件路径 (如 input.mp4)
        output_path: 输出视频文件保存路径 (默认 outputs/result_<原文件名>.mp4)
        checkpoint: 模型权重文件路径
        device_name: 运算设备 ('cuda' 或 'cpu')
        verbose: 是否持续打印每帧耗时与检测目标日志
        show: 是否在处理过程中实时弹出 OpenCV 窗口预览
    """
    if checkpoint:
        _load_model(checkpoint, device_name=device_name)
    else:
        _load_model(device_name=device_name)

    out_device = next(_get_model().parameters()).device
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return

    orig_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    out = (
        Path(output_path)
        if output_path
        else output_dir / f"result_{Path(video_path).stem}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_video = out.parent / f".temp_{out.stem}_{int(time.time() * 1000)}.mp4"

    fourcc = cv.VideoWriter.fourcc(*"mp4v")
    writer = cv.VideoWriter(str(temp_video), fourcc, fps, (orig_w, orig_h))

    logger.info(
        f"开始处理视频: {video_path} (分辨率: {orig_w}x{orig_h}, FPS: {fps:.1f}, 总帧数: {total_frames}) -> {out}"
    )

    scale_x = orig_w / IMG_W
    scale_y = orig_h / IMG_H

    if show:
        cv.namedWindow("YOLOv3 Video Processing", cv.WINDOW_NORMAL)

    frame_idx = 0
    t_start = time.perf_counter()

    with tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="Processing Video",
        unit="frame",
    ) as pbar:
        while True:
            t0 = time.perf_counter()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            input_tensor = image_to_tensor_gpu(frame_bgr, IMG_W, IMG_H, out_device)
            results = detect(input_tensor)

            draw_detections_cv(frame_bgr, results, scale_x=scale_x, scale_y=scale_y)
            writer.write(frame_bgr)

            elapsed_s = time.perf_counter() - t0
            pbar.update(1)

            if verbose:
                logger.info(
                    f"[帧 {frame_idx}/{total_frames}] {_format_detection_log(results, elapsed_s)}"
                )

            if show:
                cv.imshow("YOLOv3 Video Processing", frame_bgr)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    logger.info("用户按 'q' 键提前结束视频处理。")
                    break

    cap.release()
    writer.release()
    if show:
        cv.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    # 将原视频的音频流合入新生成的视频中
    _merge_audio(temp_video, video_path, out)

    logger.info(
        f"视频处理完成！共处理 {frame_idx} 帧，耗时 {total_time:.2f}s (平均 {avg_fps:.1f} FPS)，结果已保存至: {out}"
    )
