import cv2 as cv
import mss
import numpy as np
import torch

from utils.general import check_yaml, non_max_suppression
from utils.boxTool import draw_box
from models.yolo import Model
from utils.tools import multi_class_nms


model = Model(check_yaml('yolov3-tiny.yaml'))
model.load_state_dict(torch.load(r'0.4207_best_16.pth', map_location=torch.device('cpu'))['model'])
model.eval()

device = "cpu" if torch.cuda.is_available() else "cpu"
imgSize = 320


def normalizeData(images):
    images = np.expand_dims(images, axis=0)
    images = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    return images


def transport(image):
    nImage = cv.resize(image, (imgSize, imgSize), interpolation=cv.INTER_AREA)

    image = cv.cvtColor(nImage, cv.COLOR_BGR2RGB)
    image = normalizeData(image)
    _input = torch.tensor(image, dtype=torch.float32)
    return nImage, _input


def detect(image, x):
    outputs = model(x)
    results = non_max_suppression(outputs, agnostic=False, max_det=300)
    draw_box(image, results[0])


if __name__ == "__main__":
    is_cap = True
    is_img = False
    is_screenshot = False

    with torch.no_grad():
        if is_cap:
            cap = cv.VideoCapture(0)
            # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
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
