import cv2 as cv
import numpy as np
import torch

from config.yolov3 import CONF
from nets.yolo import YoloBody
from utils.tools import nms, buildBox

model = YoloBody(anchors_mask=CONF.anchors_mask, num_classes=80).to(CONF.device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=CONF.device)['model'])
model.eval()
def draw(img, boxes, scores, labels):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
        cv.putText(img, f'{scores[i]:.2f} {CONF.class_name[labels[i]]}', (int(x1), int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)

def process(img, input):
    with torch.no_grad():
        output = model(input)

    all_boxes, all_scores, all_labels = [], [], []

    for i in range(len(output)):
        S = CONF.feature_map[i]
        stride = CONF.sample_ratio[i]

        prediction = output[i].squeeze().view(3, 85, S, S).permute(0, 2, 3, 1)
        anchors = torch.tensor(CONF.anchors, device=CONF.device)

        boxes, scores, labels = buildBox(i, S, stride, prediction, anchors, CONF.anchors_mask)
        
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend(labels)

    # 拼接所有层输出
    if not all_boxes:
        return
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # NMS 按类别分别处理（或用 batched_nms）
    keep = nms(boxes, scores, iou_threshold=0.45)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # draw
    draw(img, boxes, scores, labels)

def transport(img, to_tensor=True):
    if to_tensor:
        img = cv.resize(img, (CONF.imgsize, CONF.imgsize))
        input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        input = np.transpose(np.array(input / 255.0, dtype=np.float32), (2, 0, 1))
        input = torch.tensor(input).unsqueeze(0).to(torch.float32).to(CONF.device)
    return img, input

if __name__ == '__main__':
    is_cap = False

    if is_cap:
        cap = cv.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                    print("无法获取帧！")
                    break
            img, input = transport(img, to_tensor=True) # to tensor
            process(img, input) # predict
            cv.namedWindow('Camera', cv.WINDOW_NORMAL)
            cv.imshow('Camera', img)
            if cv.waitKey(1) == ord('q'):
                break
        # 释放资源
        cap.release()
        cv.destroyAllWindows()
    else:
        test_img = "img/street.jpg"
        img = cv.imread(test_img)
        img, input = transport(img, to_tensor=True) # to tensor
        process(img, input) # predict
        cv.imwrite('output.jpg', img)