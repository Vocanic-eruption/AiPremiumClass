# %%
import torch
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量
from torchvision.datasets import KMNIST           # 导入KMNIST数据集

# %%
# 加载数据集 train
train_data = KMNIST(root='./KMNIST_data', train=True, download=True, 
                          transform=ToTensor())
test_data = KMNIST(root='./KMNIST_data', train=False, download=True,
                         transform=ToTensor())

# %%
train_data[1]  # 返回一个元组，第一个元素是图像数据，第二个元素是标签
train_data[1][0].shape  # 图像数据(1个颜色通道,图像高度,图像宽度)
train_data[1][0].reshape(-1).shape  # 将图像数据展平为一维张量

# %%
# 查看图像数据，手写数字
import matplotlib.pyplot as plt
img, label = train_data[4]
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.show()

# %%
# # 查看标签数量
# labels = set([clz for img,clz in train_data])
# labels

# %%
# 构建模型
import torch
import torch.nn as nn
from torch.utils.data import DataLoader  # 数据加载器
#### 隐藏层
# 线性层
linear = nn.Linear(in_features=784, out_features=128, bias=True)
# 激活函数
act = nn.Sigmoid()

### 输出层
linear2 = nn.Linear(in_features=128, out_features=10, bias=True)


# 模拟输入
x = torch.randn(10, 784)
out = linear(x)
# print(out)
out2 = act(out)
# print(out2)
out3 = linear2(out2)

softmax = nn.Softmax(dim=1)
final = softmax(out3)
print(final)

# %%
model = nn.Sequential(
    nn.Linear(784, 64), # 隐藏层
    nn.Sigmoid(),
    nn.Linear(64, 64), # 隐藏层
    nn.Sigmoid(),
    nn.Linear(64, 10) # 输出层
)

# %%
# 定义超参数
LR = 1e-3
epochs = 30
BATCH_SIZE = 128

# %%
trian_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle=True表示打乱数据

# %%
# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 优化器（模型参数更新）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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


