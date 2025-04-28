import json

import cv2 as cv
import numpy as np
import torch

from nets.yolo import YoloBody
from utils.tools import buildBox, multi_class_nms

with open("config/datasetParameter.json", "r", encoding="utf-8") as f:
    datasetConfig = json.load(f)

class_name = datasetConfig["class_name"]

device = "cpu" if torch.cuda.is_available() else "cpu"
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

model = YoloBody().to(device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=device)["model"])
model.eval()


def draw(img, boxes, scores, labels):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = boxes[i]
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

        prediction = output.view(-1, 3, 85, S, S).permute(0, 1, 3, 4, 2)
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


def transport(img, to_tensor=True):
    if to_tensor:
        img = cv.resize(img, (416, 416), interpolation=cv.INTER_AREA)
        _input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        _input = np.transpose(np.array(_input / 255.0, dtype=np.float32), (2, 0, 1))
        _input = torch.tensor(_input).unsqueeze(0).to(torch.float32).to(device)
    return img, _input


if __name__ == "__main__":
    is_cap = False

    if is_cap:
        cap = cv.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                print("无法获取帧！")
                break
            img, __input = transport(img, to_tensor=True)  # to tensor
            process(img, __input)  # predict
            cv.namedWindow("Camera", cv.WINDOW_NORMAL)
            cv.imshow("Camera", img)
            if cv.waitKey(1) == ord("q"):
                break
        # 释放资源
        cap.release()
        cv.destroyAllWindows()
    else:
        test_img = r"img/street.jpg"
        img = cv.imread(test_img)
        img, __input = transport(img, to_tensor=True)  # to tensor
        process(img, __input)  # predict
        cv.imwrite("output.jpg", img)
