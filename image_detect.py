import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import _get_model, _load_model, class_names, detect
from utils.transforms import image_transform

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
    drawer = ImageDraw.ImageDraw(resized_image)

    for result in results:
        score = float(result[4])
        class_id = int(result[5])
        label_text = f"{class_names[class_id]} {score:.2f}"
        x_min, y_min, x_max, y_max = list(map(int, result[:4]))
        x_min = max(0, min(x_min, original_size[0]))
        y_min = max(0, min(y_min, original_size[1]))
        x_max = max(0, min(x_max, original_size[0]))
        y_max = max(0, min(y_max, original_size[1]))
        drawer.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
        drawer.text((x_min, y_min - 15), label_text, fill="red")

    out = (
        Path(output_path)
        if output_path
        else output_dir / f"result_{Path(image_path).stem}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    resized_image.save(out)
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
