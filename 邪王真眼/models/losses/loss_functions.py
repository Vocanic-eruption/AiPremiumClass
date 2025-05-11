import torch.nn as nn

from enum import Enum
from typing import Dict


class LossFunctionType(Enum):
    CROSS_ENTROPY     = "cross_entropy"
    BCE_WITH_LOGITS   = "bce_with_logits"
    MSE               = "mse"

    MY_CROSS_ENTROPY  = "my_cross_entropy"


def get_loss_function(cfg: Dict[str, dict]):
    loss_function = cfg.get('loss_function')
    if loss_function == LossFunctionType.CROSS_ENTROPY:
        ignore_index = cfg.get('ignore_index', -100)
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_function == LossFunctionType.BCE_WITH_LOGITS:
        return nn.BCEWithLogitsLoss()
    elif loss_function == LossFunctionType.MSE:
        return nn.MSELoss()
    elif loss_function == LossFunctionType.MY_CROSS_ENTROPY:
        from .my_cross_entropy import my_cross_entropy
        return my_cross_entropy()
    
    return None
