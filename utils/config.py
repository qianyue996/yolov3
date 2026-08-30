"""全局配置常量。"""

# 输入图像分辨率
IMG_W = 416
IMG_H = 416

# 图像归一化参数（基于训练集统计）
NORMALIZE_MEAN = (0.4711, 0.4475, 0.4080)
NORMALIZE_STD = (0.2378, 0.2329, 0.2361)

# 默认类别配置文件路径
DEFAULT_CLASSES_PATH = "data/coco_names.yaml"

# 支持的媒体文件扩展名
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".ts",
    ".mpg",
    ".mpeg",
}

# YOLOv3 默认 Anchor（9 个尺寸）与掩码（按特征图从粗到细：13x13, 26x26, 52x52）
DEFAULT_ANCHORS = [
    [10, 13],
    [16, 30],
    [33, 23],
    [30, 61],
    [62, 45],
    [59, 119],
    [116, 90],
    [156, 198],
    [373, 326],
]
DEFAULT_ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# YOLOv3-Tiny 默认 Anchor（6 个尺寸）与掩码（按特征图：13x13, 26x26）
TINY_ANCHORS = [
    [10, 14],
    [23, 27],
    [37, 58],
    [81, 82],
    [135, 169],
    [344, 319],
]
TINY_ANCHORS_MASK = [[3, 4, 5], [0, 1, 2]]
