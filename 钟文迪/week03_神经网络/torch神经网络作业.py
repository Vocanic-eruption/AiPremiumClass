import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor
from torchvision.datasets import KMNIST
from  torch.utils.data import DataLoader

# 定义超参数
LR = 1e-2
epochs = 30
BATCH_SIZE = 128

# 数据集加载(KMNIST数据集)
train_data = KMNIST(root = './kmnist_data', train = True, download = True, transform = ToTensor())
test_data = KMNIST(root = './kmnist_data', train = False, download = True, transform = ToTensor())

train_d1 = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)


# 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10)
)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr = LR)

# 训练模型
for epoch in range(epochs):
    # 提取训练数据
    for data, target in train_d1:
        # 向前运算
        oupput = model(data.reshape(-1, 784))
        # 计算损失
        loss = loss_fn(oupput, target)
        # 反向传播
        optimizer.zero_grad() # 所有参数梯度清零
        loss.backward()  # 计算梯度
        optimizer.step() # 更新参数

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
