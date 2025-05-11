import torch
import torch.nn as nn


class MyBatchNorm(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            dims = [0] + list(range(2, x.dim()))
            batch_mean = x.mean(dim=dims)
            batch_var  = x.var(dim=dims, unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean.detach(), alpha=self.momentum)
            self.running_var .mul_(1 - self.momentum).add_(batch_var .detach(), alpha=self.momentum)
            self.num_batches_tracked += 1
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        shape = [1, -1] + [1] * (x.dim() - 2)
        mean = mean.view(*shape)
        var  = var .view(*shape)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.weight is not None:
            weight = self.weight.view(*shape)
            bias   = self.bias  .view(*shape)
            x_norm = x_norm * weight + bias

        return x_norm
