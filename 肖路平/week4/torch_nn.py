import torch
import torch.nn as nn

# 定义模型
#逐步增加神经网络内容
class TorchNN(nn.Module):#TorchNN 继承nn.Module 
    def __init__(self):# self 新创建模型对象
        super().__init__()

        self.linear1 = nn.Linear(784,512)
        self.bn1=nn.BatchNorm1d(512)#添加归一化
        self.linear2 = nn.Linear(512,512)
        self.bn2=nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512,10)
        self.drop = nn.Dropout(p=0.5)#正则化，过滤失活神经元
        self.act1 = nn.ReLU()
    # forward 前向运算
    def forward(self,input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act1(out)
        out = self.drop(out)
        final = self.linear3(out)
        return final

if __name__ == '__main__':#不会在外部调用时使用
        model = TorchNN() # 创建模型对象

        input_data = torch.randn((10,784))
        final = model(input_data)
        print(final.shape)
        
 