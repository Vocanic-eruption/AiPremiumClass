# %%
import torch
from torchvision.datasets import KMNIST
from torchvision.transforms.v2 import ToTensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# %% [markdown]
# 加载数据集

# %%
train_dataset = KMNIST(root='./data',           # 数据存储路径
    train=True,              # True为训练集，False为测试集
    download=True,           # 自动下载数据集
    transform=ToTensor()      # 应用预处理
)
test_dataset=KMNIST(root='./data',train=False,download=True,transform=ToTensor())


# %%
test_dataset

# %% [markdown]
# 超参数定义

# %%
LR=1e-3
epochs=20
Batch_size=100

# %% [markdown]
# 使用DataLoader分割数据

# %%
train_dl = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

# %% [markdown]
# 定义自己的训练网络，稍微改了一下，没有单纯用nn.Sequential

# %%
class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        #在中间多加一个隐藏层
        self.linear1=nn.Linear(784,64)
        self.Relu=nn.ReLU()
        self.linear2=nn.Linear(64,128)
        self.Relu2=nn.ReLU()
        self.linear3=nn.Linear(128,10)
        self.init_param()
    #参数初始化
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                #xavier_uniform_参数初始化
                nn.init.xavier_uniform_(m.weight)
                # 初始化偏置 b
                nn.init.zeros_(m.bias)
    def forward(self,X):
        return self.linear3(self.Relu2(self.linear2(self.Relu(self.linear1(X)))))



# %%
model=Mynet()
optimizer=torch.optim.SGD(model.parameters(),lr=LR)
print(model.parameters())
#打印训练参数
for name, param in model.named_parameters():
    print(f"参数名称: {name}, 参数形状: {param.shape}")
            

# %%
# 损失函数&优化器
loss_fn=nn.CrossEntropyLoss()
# 优化器（模型参数更新)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% [markdown]
# 训练过程

# %%
for epoch in range(epochs):
    for X,y in train_dl:
        #实现了forward可以直接前向传播
        output=model(X.reshape(-1,784))
        #损失函数计算
        loss=loss_fn(output,y)
        #反向传播
        optimizer.zero_grad()
        #计算梯度
        loss.backward()
        #参数更新
        optimizer.step()
    print(f'Epoch:{epoch} Loss: {loss.item()}')

# %% [markdown]
# 测试环节

# %%
test_dl=DataLoader(test_dataset,batch_size=Batch_size)
correct = 0
total = 0
with torch.no_grad():
    for X,y in test_dl:
        output=model(X.reshape(-1,784))
        _,predicted=torch.max(output,1)#返回每行的最大值与索引
        total+=y.size(0)
        correct+=(predicted==y).sum().item()
print(f'Accuracy:{correct/total*100}%')

# %%
predicted


