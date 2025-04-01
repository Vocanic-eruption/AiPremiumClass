import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import DataLoader
# 初始化超参数
LR = 1e-5
epochs = 100
BATCH_SIZE = 256
# 下载训练和测试数据
train_data = FashionMNIST(root = "./trainData",train=True,download=True,transform=ToTensor())
test_data = FashionMNIST(root = "./testData",train=False,download=True,transform=ToTensor())
# 分批切割处理训练数据
train_dataLoader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
# 定义训练模型
model = nn.Sequential(nn.Linear(784,512),nn.Sigmoid(),nn.Linear(512,256),nn.Sigmoid(),nn.Linear(256,128))
# 损失函数
loss_fn = nn.CrossEntropyLoss()
optimier = torch.optim.sgd(model.parameters,lr = LR)
# 训练数据
for epoch in range(epochs):
    for data,target in train_dataLoader:
        output = model(data.reshape(-1,784))
        loss = loss_fn(output,target)
        optimier.zero_grad()
        loss.backward()
        optimier.step()
    print(f"epoch:{epoch};Loss: {loss.item()}")
