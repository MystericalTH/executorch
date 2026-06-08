import torch


class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul3.default(a)
