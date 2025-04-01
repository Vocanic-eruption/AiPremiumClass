# %%
# 导入必要包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader  # 数据加载器

# %%
# 定义超参数
LR = 1e-3
epochs = 20
BATCH_SIZE = 128

# %%
# 数据集加载
#28x28 grayscale, 70,000 images
train_data = KMNIST(root='./kmnist_data', train=True, download=True, 
                          transform=ToTensor())
test_data = KMNIST(root='./kmnist_data', train=False, download=True,
                         transform=ToTensor())

# %%
trian_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# %%
# 定义模型
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.Sigmoid(),
    nn.Linear(128, 10)
)

# %%
# 损失函数&优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 优化器（模型参数更新）
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %%
for epoch in range(epochs):
    # 提取训练数据
    for data, target in trian_dl:
        # 前向运算
        output = model(data.reshape(-1, 784))
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        optimizer.zero_grad()  # 所有参数梯度清零
        loss.backward()     # 计算梯度（参数.grad）
        optimizer.step()    # 更新参数

    print(f'Epoch:{epoch} Loss: {loss.item()}')

# %%
# 测试
test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)

correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for data, target in test_dl:
        output = model(data.reshape(-1, 784))
        _, predicted = torch.max(output, 1)  # 返回每行最大值和索引
        total += target.size(0)  # size(0) 等效 shape[0]
        correct += (predicted == target).sum().item()

print(f'Accuracy: {correct/total*100}%')

# %%



