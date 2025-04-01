import torch
import torch.nn as nn

class TorchNN(nn.Module):
    # 初始化
    def __init__(self): # 初始化父类
        super().__init__()

        self.linear1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 10)
        self.drop = nn.Dropout(0.3) # 0.3的概率丢弃
        self.act = nn.ReLU()

    # 前向传播 (nn.module方法重写)
    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        fianl = self.linear3(out)
        return fianl

if __name__ == '__main__':
    model = TorchNN() #
    print(model)

    input_data = torch.randn(10, 784)# 10张图片，每张图片784个像素
    final = model(input_data)
    print(final.shape) # torch.Size([10, 10])