import cv2 as cv
import json
import yaml

with open('config/datasets.yaml', encoding="ascii", errors="ignore")as f:
    cfg = yaml.safe_load(f)
class_names = cfg['voc']['class_name']


def draw_box(image, boxes, scores, labels):
    if scores is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
            cv.putText(image, f"{scores[i].item():.2f} {class_names[labels[i]]}", (int(x1), int(y1) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
