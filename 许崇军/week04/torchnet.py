import torch
import torch.nn as nn

class TorchNN(nn.Module):
    # 初始化
    def __init__(self, use_batchnorm=True, use_dropout=True):  
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.linear1 = nn.Linear(4096, 2048)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(2048)   # 归一化
        self.act = nn.ReLU()
        if use_dropout:
            self.drop = nn.Dropout(p=0.3)     # 正则化

        self.linear2 = nn.Linear(2048, 1024)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(1024)   # 归一化
        if use_dropout:
            self.drop = nn.Dropout(p=0.3)     # 正则化

        self.linear3 = nn.Linear(1024, 40)

    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        if self.use_batchnorm:
            out = self.bn1(out)  # 归一化
        out = self.act(out)     # 激活
        if self.use_dropout:
            out = self.drop(out) # 正则化

        out = self.linear2(out)
        if self.use_batchnorm:
            out = self.bn2(out)   # 归一化
        out = self.act(out)     # 激活
        if self.use_dropout:
            out = self.drop(out) # 正则化

        final = self.linear3(out) # shape

        return torch.softmax(final, dim=-1)

if __name__ == '__main__':
    # 测试代码
    print('模型测试')
    model = TorchNN()  # 创建模型对象

    input_data = torch.randn((10, 4096))
    final = model(input_data)
    print(final.shape)