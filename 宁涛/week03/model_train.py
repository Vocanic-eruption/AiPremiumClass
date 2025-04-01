import torch.nn as nn
import torch
from torchvision.datasets import KMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def data_train_test_process():
    train_data = KMNIST(root="./data",
                        train=True,
                        transform=transforms.Compose([transforms.ToTensor()]),
                        download=True)
    test_data = KMNIST(root="./data",
                       train=False,
                       transform=transforms.Compose([transforms.ToTensor()]),
                       download=True)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=0)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=0)
    return train_dataloader, test_dataloader


def train_model(train_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = 1e-3
    epochs = 20
    # epochs = 30

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    model.to(device)

    # 损失函数&优化器
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 优化器（模型参数更新）
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    for epoch in range(epochs):
        # 提取训练数据
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)
            # 前向运算
            output = model(data.reshape(-1, 784))
            # 计算损失
            loss = loss_fn(output, target)
            # 反向传播
            optimizer.zero_grad()  # 所有参数梯度清零
            loss.backward()  # 计算梯度（参数.grad）
            optimizer.step()  # 更新参数

        print(f'Epoch: {epoch} Loss: {loss.item()}')

    # 测试
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data.reshape(-1, 784))
            _, predicted = torch.max(output, 1)  # 返回每行最大值和索引
            total += target.size(0)  # size(0) 等效 shape[0]
            correct += (predicted == target).sum().item()

    print(f'Accuracy: {correct / total * 100}%')


if __name__ == '__main__':
    train_dataloader, test_dataloader = data_train_test_process()
    train_model(train_dataloader, test_dataloader)
