import numpy as np

# Начальное приближение
xt = 7
# Шаг градиентного спуска
alpha = 10
# Вычисление значения x на следующем шаге
xt_next = xt - alpha * (1 / (xt + 3))

print(xt_next)
# ответ = 6

#                                                       задание 5



X = np.array([[1, 2],[4, 5]])

x_diff = (1 / (X+1))
answer = X - (10 * x_diff)
print(answer)