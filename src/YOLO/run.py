from ultralytics import YOLO
import cv2
import os
import torch
from torchvision import transforms
from PIL import Image

# Path to image folder
image_folder = "C:/Users/Admin/Pictures/Screenshots/12"

# Load a model
model = YOLO("yolov8n.pt")

# Get list of image paths
image_paths = [
    os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
]

# Run inference on each image
results = model(image_paths)


# Загрузка обученного классификатора
def load_classifier(path_to_pth_weights, device):
    model = torch.load(path_to_pth_weights, map_location=device)
    model.eval()
    model.to(device)
    return model


# Функция для классификации изображения
def classify_image(classifier, image):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = classifier(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Загрузка классификатора
path_to_weights = "C:/Users/Admin/Desktop/Coding/repos/project-1/resnet50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = load_classifier(path_to_weights, device)

# Process results list
for result, image_path in zip(results, image_paths):
    image = cv2.imread(image_path)

    for box in result.boxes.xyxy:
        xmin, ymin, xmax, ymax = box.tolist()
        cv2.rectangle(
            image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2
        )

        # Crop the image using the bounding box coordinates
        cropped_image = Image.fromarray(
            image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        )

        # Pass the cropped image to the classifier
        class_label = classify_image(classifier, cropped_image)
        if class_label == 0:
            class_label = "Aircraft"
        else:
            class_label = "Ship"

        cv2.putText(
            image,
            class_label,
            (int(xmin), int(ymin) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
