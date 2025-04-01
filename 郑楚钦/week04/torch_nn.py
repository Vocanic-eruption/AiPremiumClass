import torch
import torch.nn as nn

class TorchNN(nn.Module):
    # 初始化
    def __init__(self):  # self 指代新创建模型对象
        super().__init__()

        self.linear1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 10)
        self.drop = nn.Dropout(p=0.3)
        self.act = nn.ReLU()

    # forward 前向运算 (nn.Module方法重写)
    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        final = self.linear3(out) # shape

        return final

if __name__ == '__main__':
    # 测试代码
    print('模型测试')
    model = TorchNN()  # 创建模型对象

    input_data = torch.randn((10, 784))
    final = model(input_data)
    print(final.shape)
