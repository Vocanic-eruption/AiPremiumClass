import torch
import torch.nn as nn

from torch import Tensor


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y + self.bias
        return y
