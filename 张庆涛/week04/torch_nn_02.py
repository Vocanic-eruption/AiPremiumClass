import torch 
import torch.nn as nn

# 相当于定义一个模型
class TorchNN(nn.Module):
    # 初始化
    def __init__(self):  # self 指代新创建模型对象
      super().__init__()
      # 定义网络层 模型层数
      self.linear1 = nn.Linear(64*64,2048)
      self.bn1 = nn.BatchNorm1d(2048)
      
      
      self.linear2 = nn.Linear(2048,1024)
      self.bn2 = nn.BatchNorm1d(1024)
      
      self.linear3 = nn.Linear(1024,512)
      self.bn3 = nn.BatchNorm1d(512)
      
      self.linear4 = nn.Linear(512,512)
      self.bn4 = nn.BatchNorm1d(512)
      
      self.linear5 = nn.Linear(512,128)
      self.bn5 = nn.BatchNorm1d(128)
      
      self.linear6 = nn.Linear(128,40)
      self.dropout = nn.Dropout(p=0.3)
      
      self.act = nn.ReLU() # 激活函数

    
    def forward(self, input_tensor):
       # 向前传播，向前运算
      out = self.linear1(input_tensor)
      out = self.bn1(out)
      out = self.act(out)
      
      out = self.dropout(out)
      out = self.linear2(out)
      out = self.bn2(out)
      out = self.act(out)
      
      out = self.linear3(out)
      out = self.bn3(out)
      out = self.act(out)
      
      out = self.linear4(out)
      out = self.bn4(out)
      out = self.act(out)
      
      out = self.linear5(out)
      out = self.bn5(out)
      out = self.act(out)
      
      out = self.dropout(out)
      final = self.linear6(out)
    
      return final;

if __name__ == '__main__':
    print('模型测试')
    model = TorchNN()
    print(model)
    input_data = torch.randn((400, 64*64))
    final = model(input_data)
    print(final.shape)  