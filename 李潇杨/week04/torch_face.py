import torch
import torch.nn as nn

class Torch_face(nn.Module):
    # 初始化
    def __init__(self):  # self 指代新创建模型对象
        super().__init__()

        self.linear1 = nn.Linear(4096, 8192)
        self.bn1 = nn.BatchNorm1d(8192)
        self.linear2 = nn.Linear(8192, 16384)
        self.bn2 = nn.BatchNorm1d(16384)
        self.linear3 = nn.Linear(16384, 1024)
        self.drop = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(1024)
        self.linear4 = nn.Linear(1024, 40)
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
        out = self.linear3(out) # shape
        out = self.bn3(out)
        out = self.act(out)
        out = self.drop(out)
        return self.linear4(out)

if __name__ == '__main__':
    # 测试代码
    print('模型测试')
    model = Torch_face()  # 创建模型对象

    input_data = torch.randn((10, 4096))
    final = model(input_data)
    print(final.shape)
