# YOLOv3 (PyTorch)

纯手写复现，Darknet-53 主干 + FPN + 三路 YOLO 检测头。

## 环境

```bash
# Python 3.10 + uv 已配置，直接运行即可
uv run python <script>
```

依赖：PyTorch 2.3.0 / torchvision 0.18.0（CUDA 11.8 wheels），详见 `pyproject.toml`。

## 快速开始

### 1. 采样小规模数据集（可选）

COCO 全量 8.2 万张图，训练较慢。用分层采样器按比例切分：

```bash
# 采样 10%（8208 张，各类别比例与全集一致）
uv run label_util/stratified_sampler.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --ratio 0.1 \
    --output data/coco_train_10pct.txt

# 采样 1%
uv run label_util/stratified_sampler.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --ratio 0.01 \
    --output data/coco_train_1pct.txt
```

输出格式（每行一张图）：
```
/path/to/img.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,...
```

### 2. 训练

```bash
# 从随机权重开始训练（小数据集快速验证）
uv run train.py --data data/coco_train_1pct.txt --epochs 10 --batch-size 2 --checkpoint null

# 使用预训练权重继续训练
uv run train.py --data data/coco_train_10pct.txt --checkpoint 1000_0.2988.pth

# 直接使用 COCO JSON（无需预生成文本文件）
uv run train.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --epochs 120 --batch-size 2
```

训练日志 → `runs/<timestamp>/`（TensorBoard）
Checkpoint 每 1000 步保存至 `weights/<step>_<loss>.pth`

### 3. 图片检测

```bash
uv run image_detect.py img/street.jpg
uv run image_detect.py img/street.jpg --output result.png --checkpoint my_model.pth
```

检测结果默认保存至 `outputs/result_<原文件名>.png`

### 4. 摄像头实时检测

```bash
uv run detect.py
```

推理前需在项目根目录放置模型权重（如 `1000_0.2988.pth`）。

## 数据集格式

| 文件 | 说明 |
|------|------|
| `data/coco_names.yaml` | 80 类名称列表 |
| `data/voc_names.yaml` | VOC 20 类名称列表 |
| `data/coco_train_*.txt` | 采样生成的训练标签（分层采样工具输出） |

## 架构

```
nets/yolov3.py     YoloBody（Darknet-53 + FPN + 3 头）
utils/dataloader.py YOLODataset / CocoDataset，416×416 输入，固定 mean/std 归一化
utils/loss.py      YOLOLOSS（GIoU + BCE，含 ignore 机制）
utils/nms.py       non_max_suppression
utils/yolo_trainning.py CustomLR（warmup + cosine annealing）
label_util/        数据预处理（coco_util / voc_util / stratified_sampler）
```

## 注意事项

- 图片归一化 mean/std（`0.4711, 0.4475, 0.4080` / `0.2378, 0.2329, 0.2361`）为训练集统计值，不可更改
- 模型权重（`.pth`）不入库，训练前需放入项目根目录
- `evaluation.py` 引用了不存在的外部包，暂不可用
