# 使用 fashionmnist 数据集训练一个神经网络
# fashionmnist是一个衣服种类的数据集，一共有6w条数据
import torch
import torch.nn as nn
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader  # 数据加载器

#定义初始参数
learning_rate = 0.001
epochs = 10
batch_size = 128


# 训练数据集加载
train_data = FashionMNIST(root='./fashion_data', train=True, download=True, 
                          transform=ToTensor())
test_data = FashionMNIST(root='./fashion_data', train=False, download=True,
                         transform=ToTensor())
trian_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # shuffle=True表示打乱数据

print(train_data.data.shape) #torch.Size([60000, 28, 28])
# 模型定义
myModel = nn.Sequential(
    nn.Linear(784, 64), # 输入层到隐藏层的线性变换  784个输入层，64个隐藏层
    nn.Sigmoid(), # 隐藏层的Sigmoid激活函数
    nn.Linear(64, 10) # 隐藏层到输出层的线性变换  64个隐藏层，10个输出层
)

# 损失函数&优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 优化器(更新模型参数)
optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)

# 训练（非常的慢！！！！！）
for epoch in range(epochs):
    for data, target in trian_dl:
        # print(data.shape) #torch.Size([128, 1, 28, 28]) 表示的意思是有128张图片，每张图片大小是1*28*28。也可以理解为1,28,28 是不同的维度
        input_data = data.reshape(-1, 784) #torch.Size([128, 784]) 转换形状，变成128个样本，每个样本是一个长度为784的向量，转换目的是为了匹配模型的输入层
        # print(input_data.shape)
        output = myModel(input_data)
        loss = loss_fn(output, target) # 计算损失
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        
    # print('epoch: ', epoch, 'loss: ', loss.item())
    
    
# 结果测试
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)
correct = 0
total = 0
with torch.no_grad(): 
    for data, target in test_dl:
        output = myModel(data.reshape(-1, 784))
        _, predicted = torch.max(output, 1)  
        total += target.size(0)  # size(0)
        correct += (predicted == target).sum().item()
    print(f'Accuracy: {correct/total*100}%')



