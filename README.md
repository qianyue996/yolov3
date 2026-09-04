# YOLOv3 (PyTorch)

纯手写复现，Darknet-53 主干 + FPN + 三路 YOLO 检测头。支持数据增强、多尺度动态训练、主干分类预训练与统一推理评估。

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

### 2. Darknet-53 主干分类预训练（Mini-ImageNet 100）

在 Mini-ImageNet 100 数据集上预训练 Darknet-53 Backbone，产出纯净权重以加速检测网络收敛：

```bash
# 默认启动训练（Mini-ImageNet 100, 60000 张图, 100 个 Epoch）
uv run train.py --mode backbone

# 自定义参数启动
uv run train.py --mode backbone \
    --data-dir /mnt/ai_models/mini_imagenet100 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.05 \
    --num-workers 8
```

训练过程中验证集 Top-1 准确率创新高时，会自动导出纯净 Backbone 权重至 `model_data/darknet53_backbone_weights.pth`，可直接由 `YoloBody(pretrained=True)` 无缝加载。

### 3. YOLOv3 目标检测训练

```bash
# 从随机权重开始训练（多尺度 Letterbox + 全量数据增强）
uv run train.py --data data/coco_train_1pct.txt --epochs 10 --batch-size 2 --checkpoint null

# 使用预训练权重继续训练，并开启多尺度训练（默认尺度: 416, 448, 480, 512, 544, 576）
uv run train.py --data data/coco_train_10pct.txt --checkpoint weights/best.pth \
    --weights-dir weights/exp1

# 带有验证集并在每个 epoch 快速计算验证损失与保存最佳模型
uv run train.py --data data/coco_train_10pct.txt --val-data data/coco_val_10pct.txt --save-best

# 直接使用 COCO JSON（无需预生成文本文件）
uv run train.py \
    --annotation /mnt/ai_models/coco2014/annotations/instances_train2014.json \
    --image-root /mnt/ai_models/coco2014/train2014 \
    --val-annotation /mnt/ai_models/coco2014/annotations/instances_val2014.json \
    --val-image-root /mnt/ai_models/coco2014/val2014 \
    --epochs 120 --batch-size 2 --save-best
```

全部参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `detect` | 训练模式：`detect`（YOLO 检测）或 `backbone`（主干预训练） |
| `--data` | `coco_train.txt` | [detect] 训练文本标签文件 |
| `--annotation` / `--image-root` | 空 | [detect] 直接读 COCO JSON 训练集，优先于 `--data` |
| `--val-data` | 空 | [detect] 验证集文本标签文件 |
| `--val-annotation` / `--val-image-root` | 空 | [detect] 直接读 COCO JSON 验证集 |
| `--img-sizes` | `416,448,480,512,544,576` | [detect] 训练输入多尺度列表（传单一尺寸如 `416` 则固定尺寸） |
| `--no-augment` | 关 | 关闭训练数据增强（翻转、90°旋转、随机裁剪、色彩抖动） |
| `--batch-size` | `2` (detect) / `64` (backbone) | batch 大小 |
| `--epochs` | `120` (detect) / `100` (backbone) | 训练轮数 |
| `--lr` | `0.01` (detect) / `0.05` (backbone) | SGD 学习率 |
| `--checkpoint` | `None` | 预训练权重路径；传 `null` 从随机权重开始 |
| `--weights-dir` | `weights` / `weights/backbone` | checkpoint 输出目录 |
| `--save-epoch` | `1` | [detect] 按 epoch 保存的间隔（默认每个 epoch 保存一次，设为 0 关闭） |
| `--save-every` | `None` | [detect] 按 step 步数保存的间隔（指定后自动关闭按 epoch 保存） |
| `--save-best` | 关 | [detect] 额外保存最佳模型到 `<weights-dir>/best.pth`（按最低验证损失判定） |
| `--freeze-backbone` | 关 | [detect] 冻结 Darknet-53 主干，只训练 FPN + 检测头 |
| `--num-workers` | `4` | DataLoader 工作进程数 |
| `--log-every` | `10` | TensorBoard 标量写入间隔（步） |

训练日志 → `runs/<timestamp>/`（TensorBoard）

> **训练机制升级**：
> 1. **Letterbox 缩放与填充**：移除拉伸 Resize，采用保真缩放最长边 + 短边居中填充 114 灰边，几何标注框精确同步映射；
> 2. **数据增强链**：内置随机水平/垂直翻转、90° 整数倍旋转（面积守恒）、面积 50%~100% 随机裁剪与色彩抖动；
> 3. **多尺度训练自适应**：损失函数与评估端彻底解耦固定 416 分辨率，按 Batch 动态推导特征图步长并自适应缩放 Anchor；
> 4. **Focal Loss + 先验偏置初始化**：置信度采用 Focal Loss（$\alpha=0.75, \gamma=1.5$）与 RetinaNet 偏置初始化（$b=-4.6$），杜绝数值溢出与梯度爆炸。

### 4. 独立模型评估（mAP@0.5 / mAP@0.5:0.95）

使用标准 COCO / VOC 评测指标对模型权重进行多维度定量测试：

```bash
# 使用文本标签评估
uv run evaluate.py --checkpoint weights/best.pth --data data/coco_val_10pct.txt

# 使用 COCO JSON 评估
uv run evaluate.py --checkpoint weights/best.pth \
    --annotation /mnt/ai_models/coco2014/annotations/instances_val2014.json \
    --image-root /mnt/ai_models/coco2014/val2014
```

控制台将输出格式化的指标报告（包含每类与全类别的 Targets, Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95）。

### 5. 目标检测（图片 / 视频文件 / 摄像头 / 屏幕实时截屏）

```bash
# 1. 视频文件逐帧检测（自动读取、标注并保存新视频，附带进度条）
uv run detect.py data/video.mp4
uv run detect.py data/video.mp4 --output outputs/annotated.mp4 --checkpoint weights/best.pth
uv run detect.py data/video.mp4 --show  # --show 参数开启实时窗口预览（按 q 可提前结束）

# 2. 屏幕实时截屏检测（持续截取屏幕中心 416x416 区域并用 OpenCV 窗口展示）
uv run detect.py screen
uv run detect.py --screen --checkpoint weights/best.pth -v  # -v / --verbose 开启处理速度(FPS/ms)与目标统计日志

# 3. 摄像头实时检测（默认读取 0 号摄像头）
uv run detect.py
uv run detect.py 0 --checkpoint weights/best.pth -v

# 4. 单张图片检测
uv run detect.py img/street.jpg
uv run detect.py img/street.jpg --output result.png --checkpoint weights/best.pth -v
```

图片结果默认保存至 `outputs/result_<原文件名>.png`，视频结果默认保存至 `outputs/result_<原文件名>.mp4`。

推理前需在项目根目录放置模型权重（默认读取 `1000_0.2988.pth`）。

## 代码格式化与规范

```bash
make format      # 自动执行 ruff check --fix 与 ruff format
ruff check .     # 代码风格与类型注解规范检查
```

## 数据集格式

| 文件 | 说明 |
|------|------|
| `data/coco_names.yaml` | 80 类名称列表 |
| `data/voc_names.yaml` | VOC 20 类名称列表 |
| `data/coco_train_*.txt` | 采样生成的训练标签（分层采样工具输出） |

## 架构

```
train.py                  训练主入口（支持 detect 目标检测训练与 backbone 主干分类预训练）
detect.py                 统一目标检测入口（自动支持图片/摄像头/视频流/屏幕截图）
evaluate.py               独立模型评估与 mAP 评测工具（标准 VOC / COCO 格式）
nets/
├── yolov3.py             YoloBody（Darknet-53 + FPN + 3 检测头，置信度先验偏置初始化）
├── yolov3_tiny.py        YOLOv3Tiny 轻量版结构
└── darknet.py            DarkNet-53 主干网络
utils/
├── augment.py            数据增强模块（翻转、旋转、随机裁剪、色彩抖动）
├── transforms.py         Letterbox 灰边缩放与图像归一化管道
├── dataloader.py         YOLODataset / CocoDataset / yolo_collate_fn 多尺度组批
├── config.py             全局常量配置（IMG_W/IMG_H、归一化 mean/std、Anchor 设定）
├── decode.py             公共解码逻辑（decode_preds，供训练 loss 与推理后处理复用）
├── models.py             核心数据结构（RawTargets/TransformedBatch）与坐标转换
├── metrics.py            评估指标计算（Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95）
├── loss.py               YOLOLOSS（GIoU + 稳定版 Focal Loss + BCE 分类，多尺度自适应）
├── postprocess.py        推理后处理（模型加载、secend_stage 解码、detect 流程）
├── inference.py          推理与可视化执行（图片/视频/摄像头/屏幕截屏检测与 OpenCV 绘制）
├── nms.py                non_max_suppression（GPU 加速非极大值抑制）
├── stratified_sampler.py COCO 分层采样工具
└── __init__.py           统一导出（load_classes / set_seed / worker_init_fn 等）
```

## 性能调优

```bash
# worker 不够就加大
uv run train.py --num-workers 8

# 主进程吃满时优先加大 batch-size，减少每秒步数
uv run train.py --batch-size 16
```

已内置优化：`pin_memory` + `non_blocking` 异步传输、worker 线程上限（防止 N 个 worker × 全核 OpenMP 争抢）、TensorBoard 写入降频。

## 注意事项

- 图片归一化 mean/std（`0.4711, 0.4475, 0.4080` / `0.2378, 0.2329, 0.2361`）为训练集统计值，不可更改
- 模型权重（`.pth`）不入库；训练输出在 `--weights-dir`（默认 `weights/`）
