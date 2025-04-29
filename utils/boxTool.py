import cv2 as cv
import json

with open("config/datasetParameter.json", "r", encoding="utf-8") as f:
    config = json.load(f)

class_name = config['voc']['class_name']


def draw_box(image, boxes, scores, labels):
    if scores is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
            cv.putText(image, f"{scores[i]:.2f} {class_name[labels[i]]}", (int(x1), int(y1) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1)
