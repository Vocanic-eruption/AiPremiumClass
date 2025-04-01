import torch
from torch import optim
from torch import mode
import torch.nn as nn
from torchvision.datasets import KMNIST
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import DataLoader

# 加载训练数据
train_data = KMNIST(root='./kmnist_data',train=True,download=True,transform=ToTensor())
# 加载测试数据
test_data = KMNIST(root='./kmnist_data',train=False,download=True,transform=ToTensor())

# 定义超参数
LR = 1e-3 # 学习率
epochs = 30 # 训练轮数

# 创建数据加载器
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=False)

# 定义模型
model = nn.Sequential(
    nn.Linear(784,128),
    nn.Sigmoid(),
    nn.Linear(128,20)
)

# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(epochs):
    for images,labels in train_loader:
        # 前向传播运算
        output = model(images.view(images.size(0), -1))
        # 计算损失
        loss = loss_function(output,labels)
        # 反向传播
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
    print(f'Epoch:{epoch} Loss: {loss.item()}')
