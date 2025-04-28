import cv2 as cv
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

imgSize = 416


class YOLODataset(Dataset):
    def __init__(self):
        super(YOLODataset, self).__init__()
        with open("voc_train.txt", "r", encoding="utf-8") as f:
            self.datas = f.readlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image = cv.imread(self.datas[index].strip("\n").split(" ")[0])
        labels = np.array(
            [
                list(map(int, item.split(",").split(".")[0]))
                for item in self.datas[index].strip("\n").split(" ")[1:]
            ]
        )

        return image, labels


def normalizeData(images, labels):
    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    for i, label in enumerate(labels):
        labels[i][:, :4] = label[:, :4] / imgSize
    return images, labels


def xyxy2xywh(labels: list[np.array]):
    for i, label in enumerate(labels):
        cx = (label[:, 0] + label[:, 2]) / 2
        cy = (label[:, 1] + label[:, 3]) / 2
        w = label[:, 2] - label[:, 0]
        h = label[:, 3] - label[:, 1]
        ids = label[:, 4]
        nLabels = np.stack([cx, cy, w, h, ids], axis=-1)
        labels[i] = nLabels
    return labels


def resizeCvt(image: np.array, labels: np.array):
    im_h, im_w = image.shape[0], image.shape[1]
    scale = min(imgSize / im_h, imgSize / im_w)
    nh, nw = int(im_h * scale), int(im_w * scale)
    nx, ny = (imgSize - nw) // 2, (imgSize - nh) // 2
    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_AREA)
    nImage = np.full((imgSize, imgSize, 3), (128, 128, 128)).astype(np.uint8)
    nImage[ny : ny + nh, nx : nx + nw] = image

    labels[:, [0, 2]] = labels[:, [0, 2]] * scale + nx
    labels[:, [1, 3]] = labels[:, [1, 3]] * scale + ny

    cvtImage = cv.cvtColor(nImage, cv.COLOR_BGR2RGB)
    return cvtImage, labels


def ToTensor(images, labels):
    images = torch.tensor(images, dtype=torch.float32)
    labels = [torch.tensor(label, dtype=torch.float32) for label in labels]
    return images, labels


def chakan(image, labels):
    labels = list(labels)
    cv.namedWindow("show", cv.WINDOW_NORMAL)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    for i, label in enumerate(labels):
        x1, y1, x2, y2, id = [int(i) for i in label]
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        cv.putText(
            image,
            f"{id}",
            (int(x1), int(y1) - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            thickness=1,
        )
    cv.imshow("show", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def yolo_collate_fn(batch):
    # resize + bgr -> rgb
    images = []
    labels = []
    for image, label in batch:
        r_image, r_label = resizeCvt(image, label)
        images.append(r_image)
        labels.append(r_label)
    images = np.array(images)
    # 随机增强
    # images, labels = randomAug(images, labels)
    #
    # chakan(images[0], labels[0])
    labels = xyxy2xywh(labels)
    images, labels = normalizeData(images, labels)
    images, labels = ToTensor(images, labels)
    return images, labels


if __name__ == "__main__":
    dataset = YOLODataset()
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=yolo_collate_fn
    )
    for i, (images, bboxes) in enumerate(dataloader):
        print(images.shape, bboxes[0].shape)
