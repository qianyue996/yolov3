import cv2 as cv
import numpy as np
import torch

from nets.yolo import YoloBody
from utils.tools import multi_class_nms, buildBox
import yaml

with open('config/yolov3.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class_names = config['dataset']['class_names']

model = YoloBody(anchors_mask=config['model']['anchors_mask'], num_classes=80, pretrained=True).to(config['hardware']['device'])
model.load_state_dict(torch.load("checkpoint.pth", map_location=config['hardware']['device'])['model'])
model.eval()
def draw(img, boxes, scores, labels):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
        cv.putText(img, f'{scores[i]:.2f} {class_names[labels[i]]}', (int(x1), int(y1)-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)

def process(img, input):
    with torch.no_grad():
        output = model(input)

    all_boxes, all_scores, all_labels = [], [], []

    for i in range(len(output)):
        S = output[i].shape[2]
        stride = config['model']['stride'][i]

        prediction = output[i].squeeze().view(3, 85, S, S).permute(0, 2, 3, 1)
        anchors = torch.tensor(config['model']['anchors'], device=config['hardware']['device'])

        boxes, scores, labels = buildBox(i, S, stride, prediction, anchors, config['model']['anchors_mask'])
        
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
    boxes, scores, labels = multi_class_nms(boxes, scores, labels=labels, iou_threshold=0.5)

    # draw
    draw(img, boxes, scores, labels)

def transport(img, to_tensor=True):
    if to_tensor:
        img = cv.resize(img, (416, 416))
        input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        input = np.transpose(np.array(input / 255.0, dtype=np.float32), (2, 0, 1))
        input = torch.tensor(input).unsqueeze(0).to(torch.float32).to(config['hardware']['device'])
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
        test_img = r"img/street.jpg"
        img = cv.imread(test_img)
        img, input = transport(img, to_tensor=True) # to tensor
        process(img, input) # predict
        cv.imwrite('output.jpg', img)