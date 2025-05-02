import json

import cv2 as cv
import mss
import numpy as np
import torch

from utils.general import check_yaml
from utils.boxTool import draw_box
from models.yolo import Model
from utils.tools import multi_class_nms

with open("config/datasetParameter.json", "r", encoding="utf-8") as f:
    datasetConfig = json.load(f)


model = Model(check_yaml('yolov3-tiny.yaml'))
model.load_state_dict(torch.load('tiny_weight.pth', map_location=torch.device('cpu'))['model'])
model.eval()

nun_classes = datasetConfig["voc"]["num_class"]
class_name = datasetConfig["voc"]["class_name"]

device = "cpu" if torch.cuda.is_available() else "cpu"
imgSize = 416


def get_result(outputs, score_thresh=0.3, iou_thresh=0.45):
    boxes, scores, labels = [], [], []
    outputs = outputs.squeeze(0)
    bboxes = outputs[..., :4]

    _scores = outputs[..., 4].unsqueeze(-1) * outputs[..., 5:]
    for cls_id in range(nun_classes):
        cls_scores = _scores[..., cls_id]
        keep = cls_scores > score_thresh
        if keep.sum() == 0:
            continue
        cls_boxes = bboxes[keep]
        cls_scores = cls_scores[keep]
        cls_labels = torch.full((cls_scores.shape[0],), cls_id).to(device).long()

        boxes.append(cls_boxes)
        scores.append(cls_scores)
        labels.append(cls_labels)
    # 拼接所有层输出
    if not boxes:
        return None, None, None
    boxes = torch.cat(boxes)
    scores = torch.cat(scores)
    labels = torch.cat(labels)

    boxes, scores, labels = multi_class_nms(boxes, scores, labels, iou_thresh)

    return boxes, scores, labels


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


def detect(image, x):
    outputs = model(x)[0]
    boxes, scores, labels = get_result(outputs)
    draw_box(image, boxes, scores, labels)


if __name__ == "__main__":
    is_cap = True
    is_img = False
    is_screenshot = False

    with torch.no_grad():
        if is_cap:
            cap = cv.VideoCapture(0)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 416)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 416)
            while True:
                ret, img = cap.read()
                if not ret:
                    print("无法获取帧！")
                    break
                img, _input = transport(img)  # to tensor
                detect(img, _input)  # outputict
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
            detect(img, _input)  # outputict
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
                    detect(img, _input)
                    # 展示
                    cv.namedWindow("Crop Screenshot", cv.WINDOW_NORMAL)
                    cv.imshow("Crop Screenshot", img)
                    if cv.waitKey(1) == ord("q"):
                        break
            cv.destroyAllWindows()
