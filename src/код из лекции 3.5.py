import torch
import matplotlib.pyplot as plt

x_train = torch.rand(100) * 20.0 - 10.0
y_train = torch.sin(x_train)
noise = torch.randn(y_train.shape) / 5.0
y_train = y_train + noise

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)


class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


sine_net = SineNet(20)


def predict(net, x):
    y_pred = net.forward(x)
    return x.numpy(), y_pred.data.numpy()


optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch_index in range(2000):
    optimizer.zero_grad()
    y_pred = sine_net.forward(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

x_pred, y_pred = predict(sine_net, x_validation)
plt.plot(x_validation.numpy(), y_validation.numpy(), "o", label="Ground truth")
plt.plot(x_pred, y_pred, "o", c="r", label="Prediction")
plt.legend(loc="upper left")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()
