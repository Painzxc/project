# #                                       шаг 3
import torch
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.rcParams["figure.figsize"] = (13.0, 5.0)

# x_train = torch.rand(100)
# x_train = x_train * 20.0 - 10.0

# y_train = torch.sin(x_train)

# plt.plot(x_train.numpy(), y_train.numpy(), "o")
# plt.title("$y = sin(x)$")
# plt.show()


# # Определение класса нейронной сети SineNet
# class SineNet(torch.nn.Module):
#     def __init__(self, n_hidden_neurons):
#         super(SineNet, self).__init__()  # Вызов конструктора родительского класса
#         self.fc1 = torch.nn.Linear(
#             1, n_hidden_neurons
#         )  # Полносвязный слой с одним входом и n_hidden_neurons выходами
#         self.act1 = torch.nn.Tanh()  # Функция активации (гиперболический тангенс)
#         self.fc2 = torch.nn.Linear(
#             n_hidden_neurons, n_hidden_neurons
#         )  # Полносвязный слой с n_hidden_neurons входами и выходами
#         self.act2 = torch.nn.Tanh()  # Функция активации (гиперболический тангенс)
#         self.fc3 = torch.nn.Linear(
#             n_hidden_neurons, 1
#         )  # Полносвязный слой с n_hidden_neurons входами и одним выходом

#     def forward(self, x):
#         x = self.fc1(x)  # Прямой проход через первый полносвязный слой
#         x = self.act1(x)  # Применение функции активации
#         x = self.fc2(x)  # Прямой проход через второй полносвязный слой
#         x = self.act2(x)  # Применение функции активации
#         x = self.fc3(x)  # Прямой проход через третий полносвязный слой
#         return x


# # Создание экземпляра класса SineNet с заданным количеством скрытых нейронов
# n_hidden_neurons = int(input("Введите количество скрытых нейронов: "))
# sine_net = SineNet(n_hidden_neurons)
# print(sine_net)

#                шаг 14


import torch


def target_function(x):
    return 2**x * torch.sin(2**-x)


# Определение класса RegressionNet, являющегося подклассом torch.nn.Module
class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        # Определение слоев сети и активационных функций
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        # Прямой проход сигнала через сеть
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


# Ввод количества скрытых нейронов
n_hidden_neurons = int(input("Введите количество скрытых нейронов: "))
# Создание экземпляра модели регрессии
net = RegressionNet(n_hidden_neurons)

# Генерация данных для обучения
x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.0
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# Генерация данных для валидации
x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# Оптимизатор Adam для обновления параметров модели
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# Функция потерь для обучения модели
def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


# Обучение модели на заданном количестве эпох
for epoch_index in range(2000):
    optimizer.zero_grad()
    # Прямой проход сети на обучающих данных
    y_pred = net.forward(x_train)
    # Вычисление функции потерь
    loss_value = loss(y_pred, y_train)
    # Обратное распространение ошибки и обновление параметров модели
    loss_value.backward()
    optimizer.step()


# Функция для вычисления метрики (средней абсолютной ошибки)
def metric(pred, target):
    absolute_diff = torch.abs(pred - target)
    mean_absolute_error = absolute_diff.mean()
    return mean_absolute_error


# Вычисление метрики на валидационных данных и печать результата
print(
    "Средняя абсолютная ошибка:", metric(net.forward(x_validation), y_validation).item()
)


# Визуализация графика
plt.figure(figsize=(10, 6))
plt.scatter(x_validation.numpy(), y_validation.numpy(), label="Исходная функция")

plt.plot(
    x_validation.numpy(), y_train.detach().numpy(), color="r", label="Предсказания"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Исходная функция и предсказания нейронной сети")
plt.legend()
plt.show()
