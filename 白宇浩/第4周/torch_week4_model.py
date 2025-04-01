import torch
import torch.nn as nn

#模型的定义标准化写法
class TorcFace(nn.Module):
    #初始化
    def __init__(self):      #self新创建的模型对象
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,40),
        )

    #模型的训练标准化写法
    #前向运算（nn.Module方法重写）
    def forward(self,x):
        return self.net(x)