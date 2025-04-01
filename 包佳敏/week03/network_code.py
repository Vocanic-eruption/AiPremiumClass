#pytorch基于FashionMNIST数据集的神经网络模型
import numpy
import torch
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor #将数据转换为张量
import torch.nn as nn
from torch.utils.data import DataLoader #加载数据集

#定义超参数
epochs = 40 #训练次数
batch_size = 32 #每次训练的图片数量,提升训练速度
lr = 1e-2 #学习率

#加载数据集,download=True表示下载数据集,tain=True表示加载训练集,train=False表示加载测试集
#train_data = KMNIST(root='./data', train=True, download=True)
# test_data = FashionMNIST(root='./data', train=False, download=True)

train_data = KMNIST(root='./data', train=True, download=True, transform=ToTensor()) 
test_data = KMNIST(root='./data', train=False, download=True, transform=ToTensor())

labels = set([clz for img,clz in train_data])
print(labels)
btraindata = DataLoader(train_data, batch_size, shuffle=True) #shuffle=True表示打乱数据集

#搭建神经网络模型

#X输入 shape(60000, 784)
#隐藏层 shape(784, 64) 参数矩阵
#隐藏层 shape(64) 偏置bias
#输出层 shape(64, 10) 参数矩阵
#输出层 shape(10) 偏置bias
#Y输出 shape(60000, 10) 10个类别

#隐藏层
#线性层
linear = nn.Linear(784, 128, bias=True)
#激活函数
act = nn.Sigmoid()

#输出层
#线性层
linear2 = nn.Linear(128, 64, bias=True)
#激活函数
softmax = nn.Softmax(dim=1)

linear3 = nn.Linear(128, 10, bias=True)
#所有结构串联
model = nn.Sequential(linear, act,linear3)
print(model)#打印模型内容描述

#损失函数
loss = nn.CrossEntropyLoss() #交叉熵损失函数
#优化器：模型参数更新
optimizer = torch.optim.SGD(model.parameters(), lr)

for epoch in range(epochs): #打乱样本以及，批量样本收敛速度不及一个一个训练，所以对模型训练挑战增加，多轮次训练
    for img, label in btraindata:
        #前向传播
        out = model(img.reshape(-1, 784))
        #计算损失
        l = loss(out, label)
        #反向传播
        #梯度清零
        optimizer.zero_grad()
        #计算梯度(参数.grad)
        l.backward()
        #更新参数
        optimizer.step()
    
    print("epoch:", epoch, "loss:", l.item())


#测试
correct = 0
total = 0
with torch.no_grad(): #不计算梯度
    for img, label in DataLoader(test_data, batch_size=batch_size):
        out = model(img.reshape(-1, 784))
        _, pred = torch.max(out, 1) #返回每行最大值和索引
        total += label.size(0) #size(0)等价于shape[0]
        correct += (pred == label).sum().item()
print("accuracy:", correct/total)
#准确率提高的方法：增加轮次、增加神经元个数、增加隐藏层层数、增加隐藏层神经元个数、增加输出层神经元个数、改变激活函数、改变优化器、改变损失函数、改变模型、改变数据集      