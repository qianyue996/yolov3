import json

import cv2 as cv
import mss
import numpy as np
import torch

from nets.yolov3 import YOLOv3
from nets.yolov3_tiny import YOLOv3Tiny
from utils.dataloader import resizeCvt
from utils.tools import buildBox, multi_class_nms

with open("config/datasetParameter.json", "r", encoding="utf-8") as f:
    datasetConfig = json.load(f)

class_name = datasetConfig["voc"]["class_name"]

device = "cpu" if torch.cuda.is_available() else "cpu"
imgSize = 416
stride = [32, 16, 8]
anchors = [
    [7, 9],
    [16, 24],
    [43, 26],
    [29, 60],
    [72, 56],
    [63, 133],
    [142, 96],
    [166, 223],
    [400, 342],
]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

model = YOLOv3Tiny(num_classes=20).to(device)
model.load_state_dict(torch.load("tiny_checkpoint.pth", map_location=device)["model"])
model.eval()


def draw(img, boxes, scores, labels):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1
        )
        cv.putText(
            img,
            f"{scores[i]:.2f} {class_name[labels[i]]}",
            (int(x1), int(y1) - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            thickness=1,
        )


def process(img, _input):
    with torch.no_grad():
        outputs = model(_input)

    all_boxes, all_scores, all_labels = [], [], []

    for i, output in enumerate(outputs):
        S = output.shape[2]
        _stride = stride[i]

        prediction = output.view(-1, 3, 25, S, S).permute(0, 1, 3, 4, 2)
        _anchors = torch.tensor(anchors, device=device)

        boxes, scores, labels = buildBox(
            i, S, _stride, prediction, _anchors, anchors_mask
        )

        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend(labels)

    # 拼接所有层输出
    if not all_boxes:
        return
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # NMS 按类别分别处理
    boxes, scores, labels = multi_class_nms(
        boxes, scores, labels=labels, iou_threshold=0.5
    )

    # draw
    draw(img, boxes, scores, labels)


def normalizeData(images):
    images = np.expand_dims(images, axis=0)
    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    return images


def transport(image):
    im_h, im_w = image.shape[0], image.shape[1]
    scale = min(imgSize / im_h, imgSize / im_w)
    nh, nw = int(im_h * scale), int(im_w * scale)
    nx, ny = (imgSize - nw) // 2, (imgSize - nh) // 2
    image = cv.resize(image, (nw, nh), interpolation=cv.INTER_AREA)
    nImage = np.full((imgSize, imgSize, 3), (128, 128, 128)).astype(np.uint8)
    nImage[ny : ny + nh, nx : nx + nw] = image

    image = cv.cvtColor(nImage, cv.COLOR_BGR2RGB)
    image = normalizeData(image)
    _input = torch.tensor(image, dtype=torch.float32)
    return nImage, _input


if __name__ == "__main__":
    is_cap = False
    is_img = False
    is_screenshot = True

    if is_cap:
        cap = cv.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                print("无法获取帧！")
                break
            img, _input = transport(img)  # to tensor
            process(img, _input)  # predict
            cv.namedWindow("Camera", cv.WINDOW_NORMAL)
            cv.imshow("Camera", img)
            if cv.waitKey(1) == ord("q"):
                break
        # 释放资源
        cap.release()
        cv.destroyAllWindows()
    elif is_img:
        test_img = r"img/street.jpg"
        img = cv.imread(test_img)
        img, _input = transport(img)  # to tensor
        process(img, _input)  # predict
        cv.imwrite("output.jpg", img)
    elif is_screenshot:
        screen_width, screen_height = 2880, 1800
        size_w, size_h = 640, 640

        # 定义截取的区域
        monitor = {
            "top": screen_height // 2 - size_h // 2,  # y坐标
            "left": screen_width // 2 - size_w // 2,  # x坐标
            "width": size_w,  # 宽度
            "height": size_h,  # 高度
        }
        while True:
            with mss.mss() as sct:
                # 截图
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)[:, :, :3]  # BGRA -> BGR
                img, _input = transport(img)
                process(img, _input)
                # 展示
                cv.namedWindow("Crop Screenshot", cv.WINDOW_NORMAL)
                cv.imshow("Crop Screenshot", img)
                if cv.waitKey(1) == ord("q"):
                    break
        cv.destroyAllWindows()
