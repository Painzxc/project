# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO()  # load a pretrained YOLOv8n detection model
model.train(
    data="C:/Users/Admin/Desktop/Coding/repos/project-1/src/YOLO/custom.yaml",
    imgsz=640,
    epochs=3,
    batch=2,
)  # train the model

model.save("trained_yolov8.pt")
