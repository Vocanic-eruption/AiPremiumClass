import torch 
import torch.nn as nn 
import torchvision
from torchvision.datasets import KMNIST
from torchvision.transforms.V2  import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader


# 加载数据
train_data = KMNIST(root='./data', train=True, download=True, transform=ToTensor())
test_data = KMNIST(root='./data', train=False, download=True, transform=ToTensor())

#定义超参数
LR = 1e-3
epochs = 20
BATCH_SIZE = 128

#使用数据加载器，批量加载数据
train_dl = DataLoader(train_data , batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_data , batch_size=BATCH_SIZE, shuffle=True)

#定义模型
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    
    nn.Linear(128, 10)
)

# 新增权重初始化
def init_weights(m):
    # 只对全连接层进行初始化（其他层如BN层保持默认）
    if isinstance(m, nn.Linear):  # 判断是否为全连接层
        # Kaiming正态分布初始化（针对ReLU激活函数优化） 初始化
        nn.init.kaiming_normal_(m.weight, 
                              mode='fan_out',  # 按输出维度计算方差
                              nonlinearity='relu')  # 适配ReLU激活函数
        
        # 偏置项初始化为0（防止初始阶段引入噪声）
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 将偏置设置为常数0

# 递归应用初始化到所有子模块
model.apply(init_weights)  # ← 关键执行方法


#loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LR)

for epoch in range(epochs):
    # 提取训练数据
    for data, target in train_dl:
        # 前向运算
        output = model(data.reshape(-1, 784))
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        optimizer.zero_grad()  # 所有参数梯度清零
        loss.backward()     # 计算梯度（参数.grad）
        optimizer.step()    # 更新参数

    print(f'Epoch:{epoch} Loss: {loss.item()}')


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
