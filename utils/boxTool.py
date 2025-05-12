import cv2 as cv
import json
import yaml

with open('config/datasets.yaml', encoding="ascii", errors="ignore")as f:
    cfg = yaml.safe_load(f)
class_names = cfg['coco']['class_name']


def draw_box(image, results):
    for i, result in enumerate(results):
        if results[i] is None or len(results[i]) == 0:
            continue
        x1, y1, x2, y2, score, labels = result
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
        cv.putText(image, f"{score.item():.2f} {class_names[int(labels.item())]}", (int(x1), int(y1) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
