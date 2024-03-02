import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def evaluate(model, dataloader, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    predicted_labels = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predicts = model(images)
            loss = loss_function(predicts, labels)
            acc = calculate_accuracy(predicts, labels)
            epoch_loss += loss.item() * images.size(0)
            epoch_acc += acc.item() * images.size(0)

            predicted_labels.extend(predicts.argmax(dim=1).cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    total_loss = epoch_loss / len(dataloader.dataset)
    total_acc = epoch_acc / len(dataloader.dataset)

    return total_loss, total_acc, predicted_labels, true_labels


# Загрузка предобученной модели ResNet-34
model_resnet34 = torch.load("resnet34_best_loss.pth")
model_resnet34.eval()

# Загрузка предобученной модели ResNet-50
model_resnet50 = torch.load("resnet50_best_loss.pth")
model_resnet50.eval()

# Загрузка предобученной модели ResNet-18
model_resnet18 = torch.load("resnet18_best_loss.pth")
model_resnet18.eval()

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_resnet34.to(device)
model_resnet50.to(device)
model_resnet18.to(device)

# Подготовка тестовых данных
test_folder = "C:/Users/Admin/Desktop/Кодинг/репозитории/project/src/ship_vs_air/test"
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_dataset = ImageFolder(test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Оценка моделей на тестовом наборе
loss_function = torch.nn.CrossEntropyLoss()

# Оценка модели ResNet-34
(
    test_loss_resnet34,
    test_acc_resnet34,
    predicted_labels_resnet34,
    true_labels_resnet34,
) = evaluate(model_resnet34, test_loader, loss_function, device)

# Оценка модели ResNet-50
(
    test_loss_resnet50,
    test_acc_resnet50,
    predicted_labels_resnet50,
    true_labels_resnet50,
) = evaluate(model_resnet50, test_loader, loss_function, device)

# Оценка модели ResNet-18
(
    test_loss_resnet18,
    test_acc_resnet18,
    predicted_labels_resnet18,
    true_labels_resnet18,
) = evaluate(model_resnet18, test_loader, loss_function, device)

# Вывод результатов
print("ResNet-34 Results:")
print(f"Test Loss: {test_loss_resnet34:.3f} | Test Acc: {test_acc_resnet34:.2%}")

print("ResNet-50 Results:")
print(f"Test Loss: {test_loss_resnet50:.3f} | Test Acc: {test_acc_resnet50:.2%}")

print("ResNet-18 Results:")
print(f"Test Loss: {test_loss_resnet18:.3f} | Test Acc: {test_acc_resnet18:.2%}")


models = ["ResNet-34", "ResNet-50", "ResNet-18"]
test_loss_values = [test_loss_resnet34, test_loss_resnet50, test_loss_resnet18]
test_acc_values = [test_acc_resnet34, test_acc_resnet50, test_acc_resnet18]


table = "| Model | Test Loss | Test Accuracy |\n"
table += "| --- | --- | --- |\n"


for i in range(len(models)):
    table += f"| {models[i]} | {test_loss_values[i]:.3f} | {test_acc_values[i]:.2%} |\n"


print(table)


with open("results.md", "w") as f:
    f.write(table)
