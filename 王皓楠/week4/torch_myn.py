import torch
import numpy
import torch.nn as nn
class Torch_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1= nn.Linear(4096, 8192)
        #第一层尝试BatchNorm归一化
        self.bn1=nn.BatchNorm1d(8192)
        self.ln2=nn.Linear(8192,1024)
        #BatchNorm1d*2
        self.bn2=nn.BatchNorm1d(1024)
        self.ln3=nn.Linear(1024,40)
        self.dropout=nn.Dropout(p=0.3)
        self.ReLu=nn.ReLU()
    def forward(self,X):
        out=self.ln1(X)
        out=self.bn1(out)
        out=self.ReLu(out)
        out=self.ln2(out)
        out=self.bn2(out)
        out=self.ReLu(out)
        #第二层之后组合使用dropout Relu
        final=self.ln3(out)
        return final
if __name__ == '__main__':
    # 测试代码
    print('模型测试')
    model = Torch_nn()  # 创建模型对象
    input_data = torch.randn((10, 4096))
    final = model(input_data)
    print(final.shape)
