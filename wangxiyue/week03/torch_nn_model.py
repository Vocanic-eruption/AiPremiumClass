### pytorch 搭建神经网络模型
# 简单实现
import torch
import torch.nn as nn # 常用模块

# X 输入 shape（，784）# 784个特征
# 隐藏层 shape（784，64） # 神经元数量 ，参数矩阵 ， 线性层、 sigmoid
# 隐藏层 shape（64，） # 偏置 bias
# 输出层 shape (64,10) # 参数矩阵
# 输出层 shape (10，) # 偏置 bias
# Y 输出 shape（，10）# 10个类别


#隐藏层
linear1  = nn.Linear(in_features=784,out_features=64,bias=True)
activation =  nn.Sigmoid() # 激活函数
#输出层
linear2 = nn.Linear(in_features=64,out_features=10,bias=True)

# 模拟输入
x = torch.randn(10,784)
out1= linear1(x)
out2 =activation(out1)
out3 = linear2(out2)
softmax = nn.Softmax(dim=1)
final = softmax(out3)

# print(out3)



# 结构串联
model = nn.Sequential(
    nn.Linear(in_features=784,out_features=64,bias=True),
    nn.Sigmoid(),
    nn.Linear(in_features=64,out_features=10,bias=True),
    nn.Softmax(dim=1)
)

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数

# 优化器（模型参数更新）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)



print([p for p in model.parameters()])