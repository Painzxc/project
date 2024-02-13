#                                       шаг 3
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (13.0, 5.0)


# Определение класса нейронной сети SineNet
class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()  # Вызов конструктора родительского класса
        self.fc1 = torch.nn.Linear(
            1, n_hidden_neurons
        )  # Полносвязный слой с одним входом и n_hidden_neurons выходами
        self.act1 = torch.nn.Tanh()  # Функция активации (гиперболический тангенс)
        self.fc2 = torch.nn.Linear(
            n_hidden_neurons, n_hidden_neurons
        )  # Полносвязный слой с n_hidden_neurons входами и выходами
        self.act2 = torch.nn.Tanh()  # Функция активации (гиперболический тангенс)
        self.fc3 = torch.nn.Linear(
            n_hidden_neurons, 1
        )  # Полносвязный слой с n_hidden_neurons входами и одним выходом

    def forward(self, x):
        x = self.fc1(x)  # Прямой проход через первый полносвязный слой
        x = self.act1(x)  # Применение функции активации
        x = self.fc2(x)  # Прямой проход через второй полносвязный слой
        x = self.act2(x)  # Применение функции активации
        x = self.fc3(x)  # Прямой проход через третий полносвязный слой
        return x


# Создание экземпляра класса SineNet с заданным количеством скрытых нейронов
n_hidden_neurons = int(input("Введите количество скрытых нейронов: "))
sine_net = SineNet(n_hidden_neurons)
print(sine_net)
