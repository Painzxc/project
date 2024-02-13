import numpy as np
import torch

# Начальное приближение
xt = 7
# Шаг градиентного спуска
alpha = 10
# Вычисление значения x на следующем шаге
xt_next = xt - alpha * (1 / (xt + 3))

print(xt_next)
# ответ = 6

#                                                       задание 5


X = np.array([[1, 2], [4, 5]])

x_diff = 1 / (X + 1)
answer = X - (10 * x_diff)
print(answer)

#                                    часть 2  шаг 5


w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
w = w.to(device)
function = torch.prod(torch.log(torch.log(w + 7)))
function.backward()

#                                            шаг 7


w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)

alpha = 0.001

for _ in range(500):

    function = (w + 7).log().log().prod()
    function.backward()
    w.data -= alpha * w.grad
    w.grad.zero_()

#                                                                   шаг 9

w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)
alpha = 0.001
optimizer = torch.optim.SGD([w], lr=0.001)

for _ in range(500):

    function = (w + 7).log().log().prod()
    function.backward()
    optimizer.step()
    optimizer.zero_grad()
