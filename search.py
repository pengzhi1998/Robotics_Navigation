import torch
import torch.distributions.normal as norm
import torch.nn as nn
import numpy as np

data = torch.tensor([[0, 31], [-0.1, 32], [0.05, 33], [-0.5, -5], [0.4, 2],
              [-1, -19], [-0.8, -20], [0.7, -3], [0.8, -22], [1, -22]])
data_visibility = data[:, 0] - 1
data_reward = data[:, 1]
tem = 1
learning_rate = 5e-2 # Adam

class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.std = nn.Parameter(torch.Tensor([1]), requires_grad=True)

    def forward(self, x):
        # normal = torch.distributions.Normal(self.mean, self.std)
        normal = norm.Normal(self.mean, self.std)
        return normal.log_prob(x)

    def get_meanstd(self):
        return self.mean, self.std

Gaussian = Gaussian()
optimizer = torch.optim.Adam(Gaussian.parameters(), lr=learning_rate)
value = torch.mean(data_reward)

for i in range(50):
    loss = 0
    for i in range(len(data_visibility)):
        loss -= Gaussian(data_visibility[i]) * torch.exp((data_reward[i] - value)/tem)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(Gaussian.get_meanstd())