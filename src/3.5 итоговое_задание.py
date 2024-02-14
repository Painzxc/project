import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

wine = load_wine()


X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :13], wine.target, test_size=0.3, shuffle=True
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden_neurons):
        super(WineNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, n_hidden_neurons)
        self.act1 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


n_input = 13
n_hidden = 26
wine_net = WineNet(n_input, n_hidden)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-3)

batch_size = 13

loss_values = []

for epoch in range(2000):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), 13):
        optimizer.zero_grad()

        batch_indexes = order[start_index : start_index + batch_size]

        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        preds = wine_net.forward(x_batch)

        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    if epoch % 100 == 0:
        test_preds = wine_net.forward(X_test)
        test_preds = test_preds.argmax(dim=1)
        accuracy = (test_preds == y_test).float().mean()
        loss_values.append(loss_value.item())

        print("Epoch: {:5d}, Accuracy: {:.2f}".format(epoch, accuracy))

# Plot the loss values
plt.plot(range(len(loss_values)), loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("Training Progress")
plt.show()
