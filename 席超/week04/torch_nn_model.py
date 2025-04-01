import torch
import torch.nn as nn


class NN_Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 40)
        self.act = nn.ReLU()

    def forward(self, input):
        out = self.linear1(input)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        final = self.linear3(out)
        return final
    
class NN_Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 64)
        self.bn1 = nn.BatchNorm1d(64)   #归一化
        self.linear2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)   #归一化
        self.linear3 = nn.Linear(64, 40)
        self.act = nn.ReLU()

    def forward(self, input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        final = self.linear3(out)
        return final

class NN_Model_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 40)
        self.drop = nn.Dropout(p=0.3)   #正则化
        self.act = nn.ReLU()

    def forward(self, input):
        out = self.linear1(input)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.drop(out)
        final = self.linear3(out)
        return final



class NN_Model_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 64)
        self.bn1 = nn.BatchNorm1d(64)   #归一化
        self.linear2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)   #归一化
        self.linear3 = nn.Linear(64, 40)
        self.drop = nn.Dropout(p=0.3)   #正则化
        self.act = nn.ReLU()

    def forward(self, input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        final = self.linear3(out)
        return final