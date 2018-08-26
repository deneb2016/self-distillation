import torch
import torch.nn as nn
import math


class DropConnect(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(DropConnect, self).__init__()
        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.drop_prob = p
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        N = x.size(0)
        K = x.size(1)

        if self.training:
            w_mask = torch.rand((N, self.in_dim, self.out_dim), dtype=x.dtype, device=x.device).gt(self.drop_prob).to(x.dtype)
            b_mask = torch.rand((N, self.out_dim), dtype=x.dtype, device=x.device).gt(self.drop_prob).to(x.dtype)
            w = self.weight * w_mask
            b = self.bias * b_mask
            x = x.view(N, 1, K)
            y = torch.matmul(x, w).view(N, self.out_dim) + b
            y = y * (1.0 / (1.0 - self.drop_prob))
        else:
            y = torch.matmul(x, self.weight) + self.bias

        return y