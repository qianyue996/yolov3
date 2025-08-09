import torch
from u_yolov5.models.yolo import Model

model = Model("models/yolov3.yaml")

output = model.model(torch.randn(1, 3, 640, 640))
