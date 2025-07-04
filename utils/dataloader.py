import cv2 as cv
import json
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class YOLODataset(Dataset):
    def __init__(self, dataset_json_path: str = "voc_train.json"):
        super().__init__()
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            self.datasets = json.load(f)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        # image = cv.imread(self.datasets[index]['file_path'])
        image = [1]
        labels = []
        for label in self.datasets[index]['labels']:
            label = [float(v) for k, v in label.items()]
            labels.append(label)

        return image, labels


def rand():
    return np.random.random()


def randomAug(image, label):
    imgSize = image.shape[0]
    nImage = image.copy()
    nLabel = label.copy()
    h, w = nImage.shape[:2]

    # 随机翻转
    if rand() > 0.7:
        flip_type = np.random.choice([0, 1, -1])
        nImage = cv.flip(nImage, flip_type)
        # bbox: [x1, y1, x2, y2]
        if flip_type == 1:  # 水平翻转
            nLabel[:, [0, 2]] = w - nLabel[:, [2, 0]]
        elif flip_type == 0:  # 垂直翻转
            nLabel[:, [1, 3]] = h - nLabel[:, [3, 1]]
        elif flip_type == -1:  # 对角翻转（等于水平+垂直）
            nLabel[:, [0, 2]] = w - nLabel[:, [2, 0]]
            nLabel[:, [1, 3]] = h - nLabel[:, [3, 1]]

    # 随机缩放（resize 到一个随机尺寸后再 resize 回原尺寸）
    # if rand() > 0.5:
    #     scale = np.random.uniform(0.5, 0.9)  # 随机缩放比例
    #     new_w = int(w * scale)
    #     new_h = int(h * scale)
    #     nImage = cv.resize(nImage, (new_w, new_h))

    #     pad_w = imgSize - new_w
    #     pad_h = imgSize - new_h
    #     top = pad_h // 2
    #     bottom = pad_h - top
    #     left = pad_w // 2
    #     right = pad_w - left
    #     nImage = cv.copyMakeBorder(
    #         nImage, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=(128, 128, 128)
    #     )
    #     if nLabel is not None:
    #         nLabel[:, :4] *= scale
    #         nLabel[:, [0, 2]] += left
    #         nLabel[:, [1, 3]] += top
    # 随机旋转
    # if rand() > 0.3:
    #     angle = np.random.uniform(-15, 15)  # 在 -15 到 15 度之间随机转
    #     M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    #     nImage = cv.warpAffine(nImage, M, (w, h), borderMode=cv.BORDER_REFLECT)

    #     # 处理 bbox，同步旋转
    #     for i, label in enumerate(nLabel):
    #         x1, y1, x2, y2, _ = label

    #         # 四个角点
    #         corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    #         # 加上 1 维度，变成齐次坐标 [x, y, 1]
    #         ones = np.ones((4, 1))
    #         corners_hom = np.hstack([corners, ones])

    #         # 旋转
    #         rotated_corners = M @ corners_hom.T  # 2x4
    #         rotated_corners = rotated_corners.T  # 4x2

    #         # 获取新的 bbox：包围旋转后的角点
    #         x_coords = rotated_corners[:, 0]
    #         y_coords = rotated_corners[:, 1]
    #         new_x1, new_y1 = x_coords.min(), y_coords.min()
    #         new_x2, new_y2 = x_coords.max(), y_coords.max()

    #         # 可选：限制在图像边界内
    #         new_x1 = np.clip(new_x1, 0, w)
    #         new_y1 = np.clip(new_y1, 0, h)
    #         new_x2 = np.clip(new_x2, 0, w)
    #         new_y2 = np.clip(new_y2, 0, h)

    #         nLabel[i, :4] = new_x1, new_y1, new_x2, new_y2

    # 随机颜色增强
    strength = 0.2
    if rand() > 0.7:
        # 随机亮度（加减一个值）
        delta = np.random.uniform(-16, 16) * strength
        nImage = np.clip(nImage.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    if rand() > 0.7:
        # 随机对比度
        alpha = 1.0 + np.random.uniform(-0.2, 0.2) * strength
        mean = np.mean(nImage, axis=(0, 1), keepdims=True)
        nImage = np.clip((nImage - mean) * alpha + mean, 0, 255).astype(np.uint8)
    if rand() > 0.7:
        # 随机饱和度（转HSV改S通道）
        hsv = cv.cvtColor(nImage, cv.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = 1.0 + np.random.uniform(-0.3, 0.3) * strength
        hsv[..., 1] *= sat_scale
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        nImage = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    return nImage, nLabel


def xyxy2xywh(labels):
    if not isinstance(labels, list):
        labels = list(labels)
    labels = [
        np.stack([
            (l[:, 0] + l[:, 2]) / 2,
            (l[:, 1] + l[:, 3]) / 2,
            l[:, 2] - l[:, 0],
            l[:, 3] - l[:, 1],
            l[:, 4]
        ], axis=-1) for l in labels
    ]
    return labels


def resizeCvt(image=None, labels=None, imgSize=416):
    if isinstance(labels, list):
        labels = np.array(labels)
        
    if image is None:
        raise ValueError('image is None')
    im_h, im_w = image.shape[:2]
    scale = min(imgSize / im_h, imgSize / im_w)
    nh, nw = int(im_h * scale), int(im_w * scale)

    # 缩放图像
    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_AREA)

    # 计算 padding（上下左右）
    top = (imgSize - nh) // 2
    bottom = imgSize - nh - top
    left = (imgSize - nw) // 2
    right = imgSize - nw - left

    # 灰色填充
    nImage = cv.copyMakeBorder(image, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=(128, 128, 128))

    # 同步变换 bbox
    if labels is not None:
        labels[:, [0, 2]] = labels[:, [0, 2]] * scale + left
        labels[:, [1, 3]] = labels[:, [1, 3]] * scale + top

    # 转为 RGB
    nImage = cv.cvtColor(nImage, cv.COLOR_BGR2RGB)
    return nImage, labels


def normalizeData(images, labels):
    images = np.array(images)
    imgSize = images.shape[1]
    images = (images / 255.0).transpose(0, 3, 1, 2)
    for i, label in enumerate(labels):
        labels[i][:, :4] = label[:, :4] / imgSize
    return images, labels


def ToTensor(images, labels):
    images = torch.tensor(images, dtype=torch.float32)
    labels = [torch.tensor(label, dtype=torch.float32) for label in labels]
    return images, labels


def chakan(images, labels):
    for index, image in enumerate(images):
        cv.namedWindow("show", cv.WINDOW_NORMAL)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        for i, label in enumerate(labels[index]):
            x1, y1, x2, y2, _id = tuple(map(int, label))
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
            cv.circle(image, ((x2 + x1) // 2, (y2 + y1) // 2), 3, (0, 0, 255), -1)
            cv.putText(
                image,
                f"{_id}",
                (int(x1), int(y1) - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                thickness=1,
            )
        cv.imshow("show", image)
        cv.waitKey(0)
        cv.destroyAllWindows()


def single_chakan(image, labels):
    # 测试图像变换时使用
    cv.namedWindow("show", cv.WINDOW_NORMAL)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    for i, label in enumerate(labels):
        x1, y1, x2, y2, _id = [int(i) for i in label]
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        cv.putText(
            image,
            f"{_id}",
            (int(x1), int(y1) - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            thickness=1,
        )
    cv.imshow("show", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# class Yolo_collate_fn:
#     def __init__(self, sizes=(320, 416, 512, 608, 640), step=5):
#         self.sizes = sizes
#         # self.sizes = [416]
#         self.imgSize = np.random.choice(self.sizes)
#         self.step = step
#         self.count = 0
#     def __call__(self, batch):
#         self.count += 1
#         if self.count % self.step == 0:
#             self.imgSize = np.random.choice(self.sizes)
#         # resize + bgr -> rgb
#         images, labels = map(list, zip(*[resizeCvt(image, label, self.imgSize) for image, label in batch]))
#         # 随机增强
#         images, labels = zip(*[randomAug(image, label) for image, label in zip(images, labels)])
#         #
#         # chakan(images, labels)
#         labels = xyxy2xywh(labels)
#         images, labels = normalizeData(images, labels)
#         images, labels = ToTensor(images, labels)
#         return images, labels

# yolo_collate_fn = Yolo_collate_fn()

def yolo_collate_fn(batch):
    ...



if __name__ == "__main__":
    dataset = YOLODataset(dataset_json_path="./voc_train.json")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=yolo_collate_fn)
    for i, (images, bboxes) in enumerate(dataloader):
        print(images.shape, bboxes[0].shape)
