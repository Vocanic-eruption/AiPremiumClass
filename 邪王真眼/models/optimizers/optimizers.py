import torch.nn as nn
import torch.optim as optim

from enum import Enum
from typing import Any, Dict


class OptimizerType(Enum):
    SGD = "SGD"
    Adam = "Adam"
    RMSprop = "RMSprop"
    AdamW = "AdamW"
    
    My_SGD = "my_SGD"
    My_Adam = "my_Adam"


def get_optimizer(cfg: Dict[str, Any], model: nn.Module):
    def get_with_default(cfg, key, default_value):
        if key not in cfg:
            cfg[key] = default_value
            print(f"Warning: '{key}' not found in configuration. Using default value: {default_value}")
        return cfg.get(key, default_value)
    
    lr = get_with_default(cfg, 'lr', 0.01)
    weight_decay = get_with_default(cfg, 'weight_decay', 1e-4)

    optimizer = cfg.get('optimizer')

    if optimizer == OptimizerType.SGD:
        momentum = get_with_default(cfg, 'momentum', 0.9)
        return optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == OptimizerType.Adam:
        return optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == OptimizerType.My_SGD:
        from .my_sgd import MySGD
        return MySGD(params=model.parameters(), lr=lr)
    elif optimizer == OptimizerType.My_Adam:
        from .my_adam import MyAdam
        return MyAdam(params=model.parameters(), lr=lr)
    elif optimizer == OptimizerType.RMSprop:
        alpha = get_with_default(cfg, 'alpha', 0.99)
        eps = get_with_default(cfg, 'eps', 1e-08)
        return optim.RMSprop(params=model.parameters(), lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps)
    elif optimizer == OptimizerType.AdamW:
        betas = get_with_default(cfg, 'betas', (0.9, 0.999))
        eps = get_with_default(cfg, 'eps', 1e-08)
        return optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    
    return None
