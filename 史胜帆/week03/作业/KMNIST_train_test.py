#pytorch搭建神经网络 KMNIST数据集的训练
import torch
import torch.nn as nn
from torchvision.datasets import KMNIST
from torchvision.transforms.v2 import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader #数据加载器 batch数据组装成矩阵 批量计算
import numpy as np

# 1 参数设置
LR = 0.086
EPOCHS = 100
BATCH_SIZE = 256

# 2 数据预处理
train_data = KMNIST(root = './k_data',train = True,download = True,transform = ToTensor()) #,transform = ToTensor()
test_data = KMNIST(root = './k_data',train = False,download = True,transform = ToTensor()) #,transform = ToTensor()
train_dl = DataLoader(train_data,batch_size = BATCH_SIZE,shuffle = True)

# 得到数据之后 先看看形状和维度 和数据长什么样 便于下一步的处理
#print(train_data.shape)  #得到的数据集很可能是图片音频文字之类 无法直接查看shape
#一步一步来查看
# print(train_data)
# print(train_data[0])
# print(train_data[0][0])#知道了数据类型和形状 把它画出来具体看看
# print(set(clazz for img,clazz in train_data))  看看一共有多少类别

# img,clazz = train_data[1999] #先取出来
# plt.imshow(img,cmap = 'gray')
# plt.title(clazz)
# plt.show()   看完了 就开始转张量训练





# 3 神经网络搭建
model = nn.Sequential(  #都是对象 遵循大驼峰写法
    nn.Linear(in_features = 784,out_features = 128,bias = True),# 里面的参数名称要么不写 要么就和官方文档的命名一样
    nn.Sigmoid(),
    # nn.Linear(in_features = 128,out_features = 64),
    # nn.Sigmoid(),
    nn.Linear(in_features = 128,out_features = 10,bias = True),
    nn.Softmax()
)
print(model) #看看


# 4 损失函数定义
loss_fn = nn.CrossEntropyLoss() #交叉熵得到的是一个数值 即使有多个样本  

# 5 优化器 梯度下降 参数更新
optimizer = torch.optim.SGD(model.parameters(),lr = LR)

#print(par for par in model.parameters()) # 看看参数有哪些

# 6 模型训练
for epoch in range(EPOCHS):
    for data,target in train_dl:
    #前向运算
        y_hat = model(data.reshape(-1,784))
        #计算损失函数
        loss_val = loss_fn(y_hat,target)  #这里的损失是批量样本的损失 是一个损失矩阵

        #计算梯度 反向传播  
        optimizer.zero_grad() #梯度计算之前先清除上次梯度
        loss_val.backward()
        #梯度下降
        optimizer.step()
    print(f'epoch:{epoch},loss:{loss_val.item()}')

# 1 模型推理
test_ld = DataLoader(test_data,batch_size = BATCH_SIZE,shuffle = True)
correct = 0
total = 0

with torch.no_grad():
    for data , target in test_ld:
        y_hat = model(data.reshape(-1,784))
        _,predicted = torch.max(y_hat,dim = 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print(f'acc:{correct / total * 100}%')
