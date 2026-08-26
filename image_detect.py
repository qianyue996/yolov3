import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import _get_model, _load_model, class_names, detect
from utils.config import IMG_H, IMG_W
from utils.transforms import image_transform

try:
    _font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    _font = ImageFont.load_default()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


def run(
    image_path: str, output_path: str | None = None, checkpoint: str = "1000_0.2988.pth"
) -> None:
    _load_model(checkpoint)
    out_device = next(_get_model().parameters()).device

    pil_image = Image.open(image_path).convert("RGB")
    original_size = pil_image.size
    resized_image, input_tensor = image_transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(out_device)

    results = detect(input_tensor)

    scale_x = original_size[0] / IMG_W
    scale_y = original_size[1] / IMG_H
    drawer = ImageDraw.ImageDraw(pil_image)

    for result in results:
        score = float(result[4])
        class_id = int(result[5])
        label_text = f"{class_names[class_id]} {score:.2f}"
        x_min, y_min, x_max, y_max = map(float, result[:4])
        x_min = max(0, min(x_min * scale_x, original_size[0]))
        y_min = max(0, min(y_min * scale_y, original_size[1]))
        x_max = max(0, min(x_max * scale_x, original_size[0]))
        y_max = max(0, min(y_max * scale_y, original_size[1]))
        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
        drawer.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)

        text_x = x_min
        text_y = y_min - 20 if y_min >= 20 else y_min + 2
        text_bbox = drawer.textbbox((text_x, text_y), label_text, font=_font)
        drawer.rectangle(text_bbox, fill="red")
        drawer.text((text_x, text_y), label_text, fill="white", font=_font)

    out = (
        Path(output_path)
        if output_path
        else output_dir / f"result_{Path(image_path).stem}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out)
    print(f"检测结果已保存: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单张图片检测")
    parser.add_argument("image", help="图片路径")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（默认 outputs/result_<原文件名>.png）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="1000_0.2988.pth",
        help="模型权重路径",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"错误: 图片不存在 {args.image}")
        sys.exit(1)

    run(args.image, args.output, args.checkpoint)
