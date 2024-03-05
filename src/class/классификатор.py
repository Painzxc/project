import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import models, transforms
import numpy as np
import os


# Загрузка обученного классификатора
def load_classifier(path_to_pth_weights, device):
    model = torch.load(path_to_pth_weights, map_location=device)
    model.eval()
    model.to(device)
    return model


# Функция для классификации изображения
def classify_image(classifier, image_path):
    image = Image.open(image_path)
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


# Создание окна tkinter
root = tk.Tk()
root.title("Image Classifier")

# Загрузка классификатора
path_to_weights = "C:/Users/Admin/Desktop/Coding/repos/project/resnet50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = load_classifier(path_to_weights, device)

# Папка с изображениями
image_folder = "C:/Users/Admin/Desktop/Coding/repos/project/src/ship_vs_air/test/da"
# Список изображений в папке
image_list = os.listdir(image_folder)
# Индекс текущего изображения
current_image_index = 0


# Функция для обработки изображения
def process_image(image_path):
    class_label = classify_image(classifier, image_path)
    if class_label == 0:
        return "Aircraft"
    else:
        return "Ship"


# Функция для отображения изображения и результата классификации
def show_image(image_path):
    global panel
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    class_label = process_image(image_path)
    cv2.putText(img, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    if panel is not None:
        panel.configure(image=img)
        panel.image = img
    else:
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()


# Функция для обработки нажатий клавиш
def key(event):
    global current_image_index
    if event.keysym == "Left":
        # Увеличение индекса текущего изображения
        current_image_index -= 1
        if current_image_index >= len(image_list):
            current_image_index = 0
        # Путь к следующему изображению
        image_path = os.path.join(image_folder, image_list[current_image_index])
        # Отображение изображения
        show_image(image_path)
    elif event.keysym == "Right":
        # Увеличение индекса текущего изображения
        current_image_index += 1
        if current_image_index >= len(image_list):
            current_image_index = 0
        # Путь к следующему изображению
        image_path = os.path.join(image_folder, image_list[current_image_index])
        # Отображение изображения
        show_image(image_path)
    elif event.keysym == "q" or event.keysym == "Q":
        # Выход из программы
        root.destroy()


panel = None

# Отображение изображения и привязка к клавишам
image_path = "C:/Users/Admin/Desktop/Coding/repos/project/src/ship_vs_air/test/shipp/boat9_205.44_147.06_292.34_216.42.png"
show_image(image_path)
root.bind_all("<Key>", key)

# Запуск главного цикла tkinter
root.mainloop()
