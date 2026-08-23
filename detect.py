import cv2 as cv
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw

from utils.postprocess import _load_model, class_names, detect
from utils.transforms import image_transform

device = "cuda" if torch.cuda.is_available() else "cpu"


def camera_detect() -> None:
    cap = cv.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            logger.error("无法获取帧！")
            break
        image = Image.fromarray(img).convert("RGB")
        resized_image, input_image = image_transform(image)
        results = detect(input_image.unsqueeze(0).to(device))
        image_handler = ImageDraw.ImageDraw(resized_image)
        for result in results:
            score = float(result[4])
            class_id = int(result[5])
            label_text = f"{class_names[class_id]} {score}"
            x_min, y_min, x_max, y_max = list(map(int, result[:4]))
            text_x = x_min
            text_y = y_min - 15
            image_handler.rectangle(((x_min, y_min), (x_max, y_max)), outline="red")
            image_handler.text((text_x, text_y), label_text, fill="green")

        image = np.array(resized_image)
        cv.namedWindow("Camera", cv.WINDOW_NORMAL)
        cv.imshow("Camera", image)
        if cv.waitKey(1) == ord("q"):
            break
    # 释放资源
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    _load_model()
    camera_detect()
