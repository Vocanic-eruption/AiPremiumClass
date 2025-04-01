import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torchvision.datasets import KMNIST
import matplotlib.pyplot as plt
import ssl
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ssl._create_default_https_context = ssl._create_unverified_context

transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
# 加载数据
train_dataset = KMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = KMNIST(root='./data', train=False, transform=transform, download=True)
data, lable = train_dataset[0]
print(data.shape)
print(type(data))
img, clz = train_dataset[2]
img = img.reshape(28,28)
plt.imshow(img,cmap="gray")
plt.show()
labels = set([clz for img,clz in train_dataset])
print(labels)

# 超参数变化尝试
LR = 1e-1 #1e-2 1e-3
epoches = 20
batch_size =256 #128 512

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 此处尝试使用sigmoid，以及参数量大小当前效果相对最佳
model = nn.Sequential(
    nn.Linear(784, 128),  # 输入层到第一个隐藏层
    nn.ReLU(),         # 激活函数
    nn.Linear(128, 512),   # 第一个隐藏层到第二个隐藏层
    nn.ReLU(),         # 激活函数
    nn.Linear(512, 10)    # 第二个隐藏层到输出层
).to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(),lr=LR)

for epoch in range(epoches):
    for data, target in train_loader:
        data, target = data.reshape(-1, 784).to(device), target.to(device)
        # 前向计算
        output = model(data)
        loss = loss_fn(output, target)  # 计算损失
        # 反向传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        print(loss.item())

test_dl = DataLoader(test_dataset, batch_size=batch_size)
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dl:
        data, target = data.reshape(-1, 784).to(device), target.to(device)  # 迁移数据到 GPU

        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100*correct/total}")