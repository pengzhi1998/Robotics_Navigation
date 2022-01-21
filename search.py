import torch
import torch.distributions.normal as normal
import torch.nn as nn
import numpy as np

data = np.array([[0, 31], [-0.1, 32], [0.05, 33], [-0.5, -5], [0.4, 2],
              [-1, -19], [-0.8, -20], [0.7, -3], [0.8, -22], [1, -22]])
data_visibility = data[:, 0]
data_reward = data[:, 1]

N = normal.Normal(0, 1)

class Gaussian(nn.Module):
    def __init__(self):
        self.dis = normal.Normal()
    def forward(self):
        return


for i in range(10):
    loss = -N.log_prob(torch.tensor(data_visibility)) * torch.tensor(data_reward)
    loss.backward()



