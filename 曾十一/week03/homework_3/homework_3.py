# 导入必要包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader  # 数据加载器



#数据构建
train_data = KMNIST(root='./fashion_data', train = True, transform = ToTensor(), target_transform = ToTensor(), download = True)
test_data = KMNIST(root='./fashion_data', train = False, transform = ToTensor(), target_transform = ToTensor(), download = True)

##image, label = train_data[0]
##print(image.shape)  # 输出：torch.Size([1, 28, 28])
##print(label)
##labels = [label.item() for label in train_data.targets]
##unique_labels = set(labels)
##print("类别数量:", len(unique_labels))  # 输出：10



#超参数设置

LR = [1e-3, 1e-4, 1e-5]
epochs = 20
BATCH_SIZE = [128, 256, 512]

#训练模型
for lr in LR:
    for batch_size in BATCH_SIZE:
        #模型定义  变更神经元数量，增加隐藏层
        modle_1 = nn.Sequential(
        nn.Linear(784, 128), 
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.Linear(128, 10),
)
        #损失函数
        loss_fn = nn.CrossEntropyLoss()
        #优化器
        optimizer = optim.SGD(modle_1.parameters(), lr=lr)
        #数据加载器
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,pin_memory=True)  # shuffle=True表示打乱数据 pin_memory=True使用GPU加速
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,pin_memory=True)

        with open("training_log_3.txt", "a",encoding='utf-8') as f:
            for epoch in range(epochs) :
                for data in train_loader:
                    x, y = data
                    x = x.view(x.size(0), -1)
                    y = y.view(y.size(0))
                    y_pred = modle_1(x) 
                    loss = loss_fn(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print("lr", lr ,"batch_size", batch_size,"epoch:", epoch, "loss:", loss.item()) 
                f.write("lr: {}, batch_size: {} ,epoch: {}, loss: {}\n".format(lr ,batch_size,epoch, loss.item()))
        #测试模型
        with open("training_log_3.txt", "a",encoding='utf-8') as f:  # 追加模式写入准确率
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    x, y = data
                    x = x.view(x.size(0), -1)  #更换形状
                    y = y.view(y.size(0))
                    y_pred = modle_1(x)
                    _, predicted = torch.max(y_pred, dim = 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            print("准确率:", correct / total)
            f.write("准确率: {}\n".format(correct / total))