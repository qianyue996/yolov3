import cv2 as cv
import numpy as np
import torch
import torchvision

from config.yolov3 import CONF
from nets.yolo_copy import YOLOv3
from nets.yolo_loss import YOLOv3LOSS

loss_fn = YOLOv3LOSS()

model = YOLOv3().to(CONF.device)
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
        anchors = torch.tensor(CONF.anchors[i], device=CONF.device)

        grid_x, grid_y = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
        grid_x = grid_x.to(CONF.device).expand(3, -1, -1)
        grid_y = grid_y.to(CONF.device).expand(3, -1, -1)

        x = (prediction[..., 0].sigmoid() + grid_x) * stride
        y = (prediction[..., 1].sigmoid() + grid_y) * stride
        w = torch.exp(prediction[..., 2]) * anchors[:, 0].view(-1, 1, 1)
        h = torch.exp(prediction[..., 2]) * anchors[:, 0].view(-1, 1, 1)
        c = prediction[..., 4].sigmoid()
        _cls = prediction[..., 5:].sigmoid()

        scores = (c.unsqueeze(-1) * _cls).reshape(-1, 80)

        # x1y1x2y2
        x1 = torch.clamp(x - w / 2, min=0, max=CONF.imgsize).reshape(-1)
        y1 = torch.clamp(y - h / 2, min=0, max=CONF.imgsize).reshape(-1)
        x2 = torch.clamp(x + w / 2, min=0, max=CONF.imgsize).reshape(-1)
        y2 = torch.clamp(y + h / 2, min=0, max=CONF.imgsize).reshape(-1)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # 选出所有分数大于阈值的 box + 类别
        score_thresh = 0.4
        for cls_id in range(80):
            cls_scores = scores[:, cls_id]
            keep = cls_scores > score_thresh
            if keep.sum() == 0:
                continue
            cls_boxes = boxes[keep]
            cls_scores = cls_scores[keep]
            cls_labels = torch.full((cls_scores.shape[0],), cls_id, dtype=torch.int64, device=CONF.device)

            all_boxes.append(cls_boxes)
            all_scores.append(cls_scores)
            all_labels.append(cls_labels)

    # 拼接所有层输出
    if not all_boxes:
        return
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # NMS 按类别分别处理（或用 batched_nms）
    keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.45)
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

    test_img = r"D:\Python\datasets\coco2014\val2014\COCO_val2014_000000000241.jpg"
    img = cv.imread(test_img)

    # cap = cv.VideoCapture(0)

    if is_cap:
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
        img, input = transport(img, to_tensor=True) # to tensor
        process(img, input) # predict
        # cv.imwrite('output.jpg', img)
        cv.namedWindow('Camera', cv.WINDOW_NORMAL)
        cv.imshow('Camera', img)
        cv.waitKey(0)
        cv.destroyAllWindows()