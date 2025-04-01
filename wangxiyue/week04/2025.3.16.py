
import torch
from  torch import  nn
from torchvision.transforms import ToTensor
from torchvision import transforms , datasets
from torch.utils.data import  DataLoader

from wangxiyue.week04.TorchNeuralNetworkModule import TorchNeuralNetworkModule


## set cuda
def check_device():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'use {device} ')
    return device





########### 定义超参####
Learning_Rate  = 1e-3
Epochs = 10
Batch_Size = 30


def loadData(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomRotation(16),
        transforms.RandomAffine(7, translate=(0.11, 0.13), shear=0.16),
        transforms.ToTensor()
    ])
    train_datasets = datasets.KMNIST( root='../data',train=True,download=True,transform=transform_train)
    test_datasets = datasets.KMNIST( root='../data',train=False,download=True,transform=ToTensor())
    train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True
                             ,num_workers=8,pin_memory=True,persistent_workers = True,prefetch_factor =4)#,generator=torch.Generator(device=device))
    test_data = DataLoader(test_datasets, batch_size=batch_size
                            ,num_workers=8,pin_memory=True,persistent_workers = True,prefetch_factor = 4)#, generator=torch.Generator(device=device))
    return train_data, test_data

#数据加载
train_data, test_data = loadData(Batch_Size)

model = TorchNeuralNetworkModule()
# model.to(check_device())
print(model)

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵 损失函数
#########
# 优化器
# BGD  批量梯度下降
# SGD  随机梯度下降
# MBGD 小批量梯度下降
# momentum 梯度下降 动量参数
# AdaGrad 动态调整lr学习率
# RMSProp 加权 动态调整lr
# Adam 结合 momentum + AdaGrad 两者优势
#
optimizer =torch.optim.Adam(model.parameters(),lr = Learning_Rate)

model.train() #  将正则归一化模型生效
for epoch in range(Epochs):
    for data, target in train_data:
        #forward
        output = model(data.reshape(-1,784))
        loss = loss_fn(output,target)
        #backward
        model.zero_grad() ## 梯度清零
        loss.backward() ##
        optimizer.step() ##param update

    print(f'Epoch:{epoch} Loss:{loss.item():.6f}')

acc = 0
total = 0
model.eval()
with torch.no_grad():
    for data , target in test_data:
        output = model(data.reshape(-1,784))
        predicted = torch.max(output,1)
        total += target.size(0)
        acc += (predicted == target ).sum().item()
print(f'acc ===>>> {(acc/total * 100):.3f}')