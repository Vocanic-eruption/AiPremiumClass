import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model , self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 40)
        self.drop = nn.Dropout()  # 默认百分之五十失活
        self.relu = nn.ReLU()
        # forward 前向计算 重写

    def forward(self, input_tensor):
        out = self.fc1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc4(out)
        return out
