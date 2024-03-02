import torchvision.models as models
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def evaluate(model, dataloader, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predicts = model(images)
            loss = loss_function(predicts, labels)
            acc  = calculate_accuracy(predicts, labels)
            epoch_loss += loss.item()
            epoch_acc  += acc.item()
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, dataloader, optimizer, loss_function, device):
    epoch_acc = 0
    epoch_loss = 0
    model.train()
    for (images, labels) in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicts = model(images)
        loss = loss_function(predicts, labels)
        acc = calculate_accuracy(predicts, labels)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc  += acc.item()
    return epoch_loss / len(dataloader),  epoch_acc / len(dataloader)

def training_iteration(model_name: str):
    model = None
    if model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name.lower() == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise Exception(f"Invalid model name: {model_name}")
    return model


model=training_iteration("resnet50")
for name, param in model.named_parameters():
    param.requires_grad = False
model.fc=torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 500),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(500, 2)
)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)
epochs = 5


optimizer34=torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()

best_loss = 1000000
best_acc = 0
transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_path = r"C:/Users/new/Desktop/MO/репозиторий/project/src/ship_vs_air/train"
test_path  = r"C:/Users/new/Desktop/MO/репозиторий/project/src/ship_vs_air/test"
train_data = dataset.ImageFolder(train_path, transform)
test_data = dataset.ImageFolder(test_path, transform)
train_loader_1 = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader_1  = DataLoader(train_data, batch_size=16, shuffle=True)
best_loss = 10000000


for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader_1, optimizer34, loss_function, device)
    test_loss, test_acc   = evaluate(model, test_loader_1, loss_function, device)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')
    if test_loss < best_loss:
        torch.save(model, "resnet50_best_loss.pth")
        best_loss=test_loss
