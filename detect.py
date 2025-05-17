import cv2 as cv
import mss
import numpy as np
import torch
import yaml

from utils.general import check_yaml, non_max_suppression
from models.yolo import Model
from utils.tools import multi_class_nms


device = "cuda" if torch.cuda.is_available() else "cpu"
imgW = 640
imgH = 640
prev_boxes = []

model = torch.load(r"C:\Users\admin\Downloads-h\5.0597_best_58.pt", map_location=device)['model'].to(device)
model.eval()

with open('config/datasets.yaml', encoding="ascii", errors="ignore")as f:
    cfg = yaml.safe_load(f)
class_names = cfg['voc']['class_name']


def draw_box(image, results):
    global prev_boxes
    results = np.array(results)

    def iou(box1, box2):
        # box: [x1, y1, x2, y2]
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def match_boxes(curr_boxes, prev_boxes, iou_thresh=0.3):
        matches = []  # 每个当前框匹配的上一帧索引（或 -1 表示没找到）

        for i, curr in enumerate(curr_boxes):
            best_iou = 0
            best_idx = -1
            for j, prev in enumerate(prev_boxes):
                iou_score = iou(curr[:4], prev[:4])
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_idx = j
            if best_iou >= iou_thresh:
                matches.append((i, best_idx))  # 当前第i个框 匹配上上一帧第best_idx个
            else:
                matches.append((i, -1))  # 当前这个框是新来的，没有匹配
        return matches

    def smooth_matched(curr_boxes, prev_boxes, matches, alpha=0.6):
        prev_boxes = np.array(prev_boxes)
        smoothed = []

        for curr_idx, prev_idx in matches:
            curr = curr_boxes[curr_idx]
            if prev_idx == -1:
                # 没有匹配上，不平滑
                smoothed.append(curr)
            else:
                prev = prev_boxes[prev_idx]
                smoothed_box = alpha * curr[:4] + (1 - alpha) * prev[:4]
                smoothed.append([
                    *smoothed_box.tolist(),
                    curr[4],  # 保留当前置信度
                    curr[5]   # 保留当前类别
                ])
        return smoothed

    matches = match_boxes(results, prev_boxes)
    smoothed_boxes = smooth_matched(results, prev_boxes, matches)
    for i, result in enumerate(smoothed_boxes):
        x1, y1, x2, y2 = tuple(map(int, result[:4]))
        score, label = result[4], int(result[-1])
        cv.circle(image, ((x2 + x1) // 2, (y2 + y1) // 2), 3, (0, 0, 255), -1)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv.putText(image, f"{score:.2f} {class_names[label]}", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
    prev_boxes = smoothed_boxes


def normalizeData(images):
    images = np.expand_dims(images, axis=0)
    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    return images


def transport(image):
    nImage = cv.resize(image, (imgW, imgH), interpolation=cv.INTER_AREA)

    image = cv.cvtColor(nImage, cv.COLOR_BGR2RGB)
    image = normalizeData(image)
    _input = torch.tensor(image).to(device)
    return nImage, _input


def detect(image, x):
    outputs = model(x)
    results = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45,agnostic=False, max_det=300)
    draw_box(image, results[0])


if __name__ == "__main__":
    is_cap = True
    is_img = False
    is_screenshot = False

    with torch.no_grad():
        if is_cap:
            cap = cv.VideoCapture(0)
            # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
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
            print('successfully!!')
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
