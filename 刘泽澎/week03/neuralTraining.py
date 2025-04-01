import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.v2 import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# 下载和加载 KMNIST 数据集
# train_dataset = datasets.KMNIST(
#     root="./dataKMNIST", train=True, download=True, transform=ToTensor()
# )
# test_dataset = datasets.KMNIST(
#     root="./dataKMNIST", train=False, download=True, transform=ToTensor()
# )

train_dataset = FashionMNIST(root='./dataKMNIST', train=True, download=True, 
                          transform=ToTensor())
test_dataset = FashionMNIST(root='./dataKMNIST', train=False, download=True,
                         transform=ToTensor())
print(train_dataset)
print(test_dataset)

# 定义超参数
LR_CON = 1e-3
epochs_CON = 20
BATCH_SIZE_CON = 128

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CON, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_CON, shuffle=False)

print(train_loader)


# 定义模型
model = nn.Sequential(
    nn.Linear(784, 64),   # 将输入图像从 (batch_size, 1, 28, 28) 转换为 (batch_size, 784)
    nn.Sigmoid(), 
    nn.Linear(64, 10))

# 损失函数&优化器
loss_fn_CON = nn.CrossEntropyLoss()
# 优化器
optimizer_CON = torch.optim.SGD(model.parameters(), lr=LR_CON)  # SGD优化器

# 训练模型
for epoch in range(epochs_CON):
    for data, target in train_loader:
        # 前向运算
        output = model(data.reshape(-1, 784))
        # 计算损失
        loss = loss_fn_CON(output, target)
        # 反向传播
        optimizer_CON.zero_grad()  # 所有参数梯度清零
        loss.backward()  # 计算梯度（参数.grad）
        optimizer_CON.step()  # 更新参数
        
    print(f"Epoch:{epoch} Loss: {loss.item()}")
