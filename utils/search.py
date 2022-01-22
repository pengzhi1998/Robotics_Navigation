import torch
import torch.distributions.normal as norm
import torch.nn as nn

# data = torch.tensor([[0, 31], [-0.1, 32], [0.05, 33], [-0.5, -5], [0.4, 2],
#               [-1, -19], [-0.8, -20], [0.7, -3], [0.8, -22], [1, -22]])
# data = torch.tensor([[-1, -100], [-0.8, -109], [-0.6, -50], [-0.4, -52], [-0.2, -20],
#               [0, -19], [0.2, -2], [0.4, 3], [0.6, 3], [0.8, -10], [1, -59]])

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