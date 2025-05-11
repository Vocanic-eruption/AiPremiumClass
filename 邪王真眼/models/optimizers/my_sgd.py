import torch

from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data -= lr * p.grad
        return
