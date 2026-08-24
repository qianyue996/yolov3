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
uv run utils/stratified_sampler.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --ratio 0.1 \
    --output data/coco_train_10pct.txt

# 采样 1%
uv run utils/stratified_sampler.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --ratio 0.01 \
    --output data/coco_train_1pct.txt
```

> 也支持模块方式：`uv run -m utils.stratified_sampler`（需在项目根目录执行）。

输出格式（每行一张图）：
```
/path/to/img.jpg x_min,y_min,x_max,y_max,class_id x_min,y_min,...
```

### 2. 训练

```bash
# 从随机权重开始训练（小数据集快速验证）
uv run train.py --data data/coco_train_1pct.txt --epochs 10 --batch-size 2 --checkpoint null

# 使用预训练权重继续训练，checkpoint 存到自定义目录，每个 epoch 都保存
uv run train.py --data data/coco_train_10pct.txt --checkpoint 1000_0.2988.pth \
    --weights-dir weights/exp1

# 直接使用 COCO JSON（无需预生成文本文件）
uv run train.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --epochs 120 --batch-size 2
```

全部参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `coco_train.txt` | 文本标签文件 |
| `--annotation` / `--image-root` | 空 | 直接读 COCO JSON，优先于 `--data` |
| `--batch-size` | `2` | batch 大小 |
| `--epochs` | `120` | 训练轮数 |
| `--lr` | `0.01` | SGD 学习率 |
| `--checkpoint` | `None` | 预训练权重路径；传 `null` 从随机权重开始 |
| `--weights-dir` | `weights` | checkpoint 输出目录 |
| `--save-every` | `1000` | 每隔多少步保存一次 `<step>_<loss>.pth` |
| `--no-save-epoch` | 关 | 关闭按 epoch 保存（默认开启，存为 `epoch<N>_<loss>.pth`） |
| `--save-best` | 关 | 额外保存 avg_loss 最低的 `best.pth` |
| `--freeze-backbone` | 关 | 冻结 Darknet-53 主干，只训练 FPN + 检测头 |
| `--num-workers` | `4` | DataLoader 工作进程数 |
| `--log-every` | `10` | TensorBoard 标量写入间隔（步） |

训练日志 → `runs/<timestamp>/`（TensorBoard）

> **重要**：loss 已修复正样本分配 bug 并加入 conf 正负样本加权（pos_weight）。
> 用旧代码训练的 checkpoints 与新代码训练动态不兼容，建议重新训练。

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

推理前需在项目根目录放置模型权重（默认读取 `1000_0.2988.pth`）。

## 数据集格式

| 文件 | 说明 |
|------|------|
| `data/coco_names.yaml` | 80 类名称列表 |
| `data/voc_names.yaml` | VOC 20 类名称列表 |
| `data/coco_train_*.txt` | 采样生成的训练标签（分层采样工具输出） |

## 架构

```
nets/yolov3.py            YoloBody（Darknet-53 + FPN + 3 头）
nets/yolov3_tiny.py       YOLOv3Tiny 轻量版（独立模块，未接入训练流程）
nets/darknet.py           DarkNet-53 主干
utils/
├── config.py             常量：IMG_W/IMG_H、归一化 mean/std
├── models.py             数据类型（RawTargets/TransformedBatch）+ xyxy2xywh 坐标转换
├── loss_types.py         loss 内部数据结构（TargetBuild/PredDecode/LayerMetrics）
├── transforms.py         图像归一化/变换（TransFormer、image_transform、image_show）
├── dataloader.py         YOLODataset / CocoDataset / yolo_collate_fn
├── loss.py               YOLOLOSS（GIoU + BCE，conf 带 pos_weight，ignore 机制）
├── postprocess.py        推理后处理（模型加载、secend_stage 解码、detect 流程）
├── nms.py                non_max_suppression
├── stratified_sampler.py COCO 分层采样工具
└── __init__.py           统一导出（load_classes / set_seed / worker_init_fn 等）
```

数据流向：

```
图片+标注 → RawTargets(像素 xyxy)
    → yolo_collate_fn/TransFormer 归一化到 416×416 → TransformedBatch([0,1] xyxy)
    → YoloBody 前向 → 三层 (B,3,H,W,5+C) logit
    → YOLOLOSS：xyxy2xywh 到各层 grid → build_targets/get_ignore → GIoU+BCE
推理：YoloBody → secend_stage 解码到像素 → non_max_suppression
```

## 性能调优（GPU 训练时 CPU 单核吃满）

```bash
# worker 不够就加大
uv run train.py --num-workers 8

# 主进程吃满时优先加大 batch-size，减少每秒步数
uv run train.py --batch-size 16
```

已内置优化：`pin_memory` + `non_blocking` 异步传输、worker 线程上限
（防止 N 个 worker × 全核 OpenMP 争抢）、TensorBoard 写入降频。

定位方法：

```bash
nvidia-smi -l 1   # GPU 利用率低 = 瓶颈在 CPU 侧
htop              # worker 进程满 → 加 num-workers；主进程满 → 加大 batch-size
```

## 注意事项

- 图片归一化 mean/std（`0.4711, 0.4475, 0.4080` / `0.2378, 0.2329, 0.2361`）为训练集统计值，不可更改
- 模型权重（`.pth`）不入库；训练输出在 `--weights-dir`（默认 `weights/`），推理用的根目录权重需手动放置
- `evaluation.py` 引用了不存在的外部包，暂不可用
- 数据增强尚未实现，小数据集易过拟合，建议尽快补充验证集与增广
