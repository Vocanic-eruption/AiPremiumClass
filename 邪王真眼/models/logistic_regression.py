import torch.nn as nn

from typing import Any, Dict


class LogisticRegression(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(LogisticRegression, self).__init__()
        
        input_size = cfg["input_size"]
        num_classes = cfg["num_classes"]
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.bn(x)
        out = self.fc(x)
        return {'out': out}