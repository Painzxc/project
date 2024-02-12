import numpy as np

# Функция f(x)
def f(x):
    return np.log(x + 3)

# Начальное приближение
xt = 7
# Шаг градиентного спуска
alpha = 10
# Вычисление значения x на следующем шаге
xt_next = xt - alpha * (1 / (xt + 3))

print(xt_next)