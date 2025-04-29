import json

import cv2 as cv
import mss
import numpy as np
import torch

from utils.boxTool import draw_box

# from nets.yolov3 import YOLOv3
from nets.yolov3_tiny import YOLOv3Tiny
from utils.tools import multi_class_nms

with open("config/datasetParameter.json", "r", encoding="utf-8") as f:
    datasetConfig = json.load(f)

with open('config/model.json', 'r', encoding="utf-8") as f:
    modelConfig = json.load(f)

nun_classes = datasetConfig["voc"]['length']
class_name = datasetConfig["voc"]["class_name"]

device = "cpu" if torch.cuda.is_available() else "cpu"
imgSize = modelConfig['yolov3_tiny']['imgSize']
stride = modelConfig['yolov3_tiny']['stride']
anchor = torch.tensor(modelConfig['yolov3_tiny']['anchor']).to(device)
anchor_mask = modelConfig['yolov3_tiny']['anchor_mask']

model = YOLOv3Tiny(num_classes=nun_classes).to(device)
model.load_state_dict(torch.load("tiny_checkpoint.pth", map_location=device)["model"])
model.eval()


def get_result(outputs, score_thresh=0.3, iou_thresh=0.45):
    boxes, scores, labels = [], [], []
    for i, output in enumerate(outputs):
        S = output.shape[2]
        layer_stride = stride[i]
        output = output.view(-1, 3, 5 + nun_classes, S, S).permute(0, 1, 3, 4, 2)

        grid_x, grid_y = torch.meshgrid(
            torch.arange(S).to(device),
            torch.arange(S).to(device),
            indexing="ij",
        )
        output[..., 0] = (output[..., 0].sigmoid() + grid_x) * layer_stride
        output[..., 1] = (output[..., 1].sigmoid() + grid_y) * layer_stride
        output[..., 2] = torch.exp(output[..., 2]) * anchor[anchor_mask[i]][:, 0].view(1, -1, 1, 1)
        output[..., 3] = torch.exp(output[..., 3]) * anchor[anchor_mask[i]][:, 1].view(1, -1, 1, 1)
        bboxes = torch.stack(
            [
                output[..., 0] - output[..., 2] / 2,
                output[..., 1] - output[..., 3] / 2,
                output[..., 0] + output[..., 2] / 2,
                output[..., 1] + output[..., 3] / 2,
            ],
            dim=-1,
        )
        pos_conf = output[..., 4].sigmoid()
        cls_conf = output[..., 5:].sigmoid()
        layer_scores = pos_conf.unsqueeze(-1) * cls_conf
        for cls_id in range(nun_classes):
            cls_scores = layer_scores[..., cls_id]
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
    outputs = model(x)

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
