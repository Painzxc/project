import torch
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import torchvision.datasets

MNIST_train = torchvision.datasets.MNIST("./", download=True, train=True)
MNIST_test = torchvision.datasets.MNIST("./", download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train.dtype, y_train.dtype

X_train = X_train.float()
X_test = X_test.float()

X_train.shape, X_test.shape

y_train.shape, y_test.shape

import matplotlib.pyplot as plt

plt.imshow(X_train[0, :, :])
plt.show()
print(y_train[0])


X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])


class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x


mnist_net = MNISTNet(100)

# torch.cuda.is_available()
# !nvidia-smi
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mnist_net = mnist_net.to(device)
# list(mnist_net.parameters())

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)


batch_size = 100

test_accuracy_history = []
test_loss_history = []

# X_test = X_test.to(device)
# y_test = y_test.to(device)

for epoch in range(10):
    order = np.random.permutation(len(X_train))  # перемешивает датасет

    for start_index in range(
        0, len(X_train), batch_size
    ):  # 0, len(X_train), batch_size
        optimizer.zero_grad()

        batch_indexes = order[
            start_index : start_index + batch_size
        ]  #  это чтобы внутри 1 эпохи каждая картинка была показана всего 1 раз

        X_batch = X_train[batch_indexes]  # .to(device)
        y_batch = y_train[batch_indexes]  # .to(device)

        preds = mnist_net.forward(X_batch)

        loss_value = loss(
            preds, y_batch
        )  # считаем loss по нашим по выходу нейронки и потому что она должна предсказать
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(
        X_test
    )  # вызываем и на тестовом датасете который нейронка не видела
    test_loss_history.append(loss(test_preds, y_test).data)

    accuracy = (
        (test_preds.argmax(dim=1) == y_test).float().mean()
    )  # номер нейрона одного из 10/доля правильных ответов
    test_accuracy_history.append(accuracy)
    print(accuracy)

    plt.plot(test_accuracy_history)

    plt.plot(test_loss_history)
plt.show()


# Выберите случайное изображение из тестового набора данных
index = random.randint(0, len(X_test))
image = X_test[index]

# Измените форму изображения и выполните нормализацию
image = image.reshape(1, -1).float() / 255

# Сделайте предсказание с помощью нейронной сети
prediction = mnist_net.forward(image)

# Получите предсказанную метку
predicted_label = torch.argmax(prediction).item()

# Отобразите изображение и предсказанную метку
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title(f"Предсказанная метка: {predicted_label}")
plt.show()
