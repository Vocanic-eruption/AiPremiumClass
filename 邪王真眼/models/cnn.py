import torch
import torch.nn as nn

from typing import Any, Dict


class CNN(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(CNN, self).__init__()
        input_size  = cfg.get("input_size")
        input_channels  = cfg.get("input_channels")
        hidden_channels = cfg.get("hidden_channels", input_channels)
        num_classes = cfg.get("num_classes")

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(input_size[0]*input_size[1]*hidden_channels//16, num_classes*4)
        
        self.fc2 = nn.Linear(num_classes*4, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return {'out': x}
    