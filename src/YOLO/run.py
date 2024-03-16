from ultralytics import YOLO

# Load a model
model = YOLO("trained_yolov8.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(
    [
        "C:/Users/Admin/Pictures/корабли/ship/boat41.png",
        "C:/Users/Admin/Pictures/корабли/ship/boat42.png",
        "C:/Users/Admin/Pictures/корабли/ship/boat43.png",
        "C:/Users/Admin/Pictures/корабли/ship/boat44.png",
        "C:/Users/Admin/Pictures/корабли/ship/boat45.png",
    ]
)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
