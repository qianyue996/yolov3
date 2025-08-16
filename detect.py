import cv2 as cv
import mss
import torchvision.transforms as T
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw

from utils import non_max_suppression


device = "cuda" if torch.cuda.is_available() else "cpu"
img_w = 416
img_h = 416

model = torch.load(r"3000_0.2839.pth", map_location=device, weights_only=False)
class_names = model.class_names
anchors = model.anchors
anchors_mask = model.anchors_mask
model.eval()


resize = T.Resize((img_w, img_h))
to_tensor = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=(0.4711, 0.4475, 0.4080), std=(0.2378, 0.2329, 0.2361)),
    ]
)


def transform(image: Image.Image):
    resized_image = resize(image)
    to_tensor_image = to_tensor(resized_image)

    return resized_image, to_tensor_image


def secend_stage(outputs):
    _outputs = []
    for i, output in enumerate(outputs):
        output = output.squeeze()
        na, size_w, size_h, _ = output.shape
        xy, wh, conf = output.split((2, 2, 1 + len(class_names)), 3)
        xy = xy.sigmoid() * 2 - 0.5
        wh = (wh.sigmoid() * 2) ** 2 * torch.tensor(anchors)[anchors_mask[i]].unsqueeze(
            1
        ).unsqueeze(1)
        grid_x = (
            torch.linspace(0, size_w - 1, size_w)
            .repeat(size_h, 1)
            .repeat(int(na), 1)
            .view(xy.shape[:3])
            .type_as(xy)
        )
        grid_y = (
            torch.linspace(0, size_h - 1, size_h)
            .repeat(size_w, 1)
            .t()
            .repeat(int(na), 1)
            .view(xy.shape[:3])
            .type_as(xy)
        )
        x = (xy[..., 0] + grid_x).unsqueeze_(-1)
        y = (xy[..., 1] + grid_y).unsqueeze_(-1)
        w = wh[..., 0].unsqueeze_(-1)
        h = wh[..., 1].unsqueeze_(-1)
        c = conf.sigmoid()
        output = torch.cat([x, y, w, h, c], dim=-1).view(1, -1, len(class_names) + 5)
        _outputs.append(output)

    return torch.cat(_outputs, dim=1).squeeze()


def detect(image):
    outputs = model(image)
    outputs = secend_stage(outputs)
    results = non_max_suppression(outputs, conf_thres=0.01, iou_thres=0.45)
    return results


def camera_detect():
    cap = cv.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            logger.error("无法获取帧！")
            break
        image = Image.fromarray(img).convert("RGB")
        resized_image, input_image = transform(image)
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
    pass


def image_detect():
    test_img = r"img/street.jpg"
    image = Image.open(test_img)
    resized_image, input_image = transform(image)
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

    resized_image.save("result.png")


if __name__ == "__main__":
    camera_detect()
    # image_detect()

    # is_cap = True
    # is_img = False
    # is_screenshot = False

    # with torch.no_grad():
    #     if is_cap:
    #         cap = cv.VideoCapture(0)
    #         # cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    #         # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    #         while True:
    #             ret, img = cap.read()
    #             if not ret:
    #                 print("无法获取帧！")
    #                 break
    #             img, _input = transport(img)  # to tensor
    #             detect(img, _input)  # outputict
    #             cv.namedWindow("Camera", cv.WINDOW_NORMAL)
    #             cv.imshow("Camera", img)
    #             if cv.waitKey(1) == ord("q"):
    #                 break
    #         # 释放资源
    #         cap.release()
    #         cv.destroyAllWindows()
    #     elif is_img:
    #         test_img = r"img/street.jpg"
    #         img = cv.imread(test_img)
    #         img, _input = transport(img)  # to tensor
    #         detect(img, _input)  # outputict
    #         cv.imwrite("output.jpg", img)
    #         print("successfully!!")
    #     elif is_screenshot:
    #         screen_width, screen_height = 2880, 1800
    #         size_w, size_h = 640, 640

    #         # 定义截取的区域
    #         monitor = {
    #             "top": screen_height // 2 - size_h // 2,  # y坐标
    #             "left": screen_width // 2 - size_w // 2,  # x坐标
    #             "width": size_w,  # 宽度
    #             "height": size_h,  # 高度
    #         }
    #         while True:
    #             with mss.mss() as sct:
    #                 # 截图
    #                 screenshot = sct.grab(monitor)
    #                 img = np.array(screenshot)[:, :, :3]  # BGRA -> BGR
    #                 img, _input = transport(img)
    #                 detect(img, _input)
    #                 # 展示
    #                 cv.namedWindow("Crop Screenshot", cv.WINDOW_NORMAL)
    #                 cv.imshow("Crop Screenshot", img)
    #                 if cv.waitKey(1) == ord("q"):
    #                     break
    #         cv.destroyAllWindows()
