import torch.nn as nn
"""
    包含BatchNorm层的全神经网络
        加速模型收敛，防止过拟合
"""
class SimpleNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 2028)
        self.bn1 = nn.BatchNorm1d(2028)
        self.linear2 = nn.Linear(2028, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 40)
        self.relu = nn.ReLU()
        
    def forward(self, input_tensor):
        out1 = self.linear1(input_tensor)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.linear2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)
        final = self.linear3(out1)
        return final
    
    