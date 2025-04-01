import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms



# 数据集加载
def data_load(batch_size):
    trainData = datasets.FashionMNIST(
        root='../data',
        train=True, # 训练集
        download=True,
        transform=transforms.ToTensor()  # 原始数据转换成 张量
    )
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)

    testData = datasets.FashionMNIST(
        root='../data',
        train=False, # 测试集
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size)

    return train_loader , test_loader

def device_check():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
    print(f'use {device} ')

# 结构串联
model = nn.Sequential(
    nn.Linear(in_features=784, out_features=64, bias=True),

    nn.ReLU(),
    nn.Linear(in_features=64, out_features=64, bias=True),
    nn.Sigmoid(),
    nn.ReLU(),
    nn.LogSoftmax(),
    nn.Linear(in_features=64, out_features=64, bias=True),

    nn.Linear(in_features=64, out_features=10, bias=True),
    # nn.Softmax(dim=1)
)

# 损失函数
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数

# 优化器（模型参数更新）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


if __name__ == '__main__':

    LR = 1e-2
    EPOCHS = 20
    BATCH_SIZE = 32
    train_data, test_data = data_load(BATCH_SIZE)
    device = device_check()

    for epoch in range(EPOCHS):
        #提取训练数据
        for data, target in train_data:
            #前向计算
            output = model(data.reshape(-1,784))
            #计算损失
            loss = loss_function(output, target)
            #反向传播
            optimizer.zero_grad() #梯度参数晴空
            loss.backward() # 梯度计算
            optimizer.step() # 更新参数
            #
            print(f'Epoch : {epoch} ,  Loss : {loss.item()}')


    #测试
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_data:
            output = model(data.reshape(-1,784))
            _, predicted = torch.max(output.data, 1) # 返回每行最大
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f'Accuracy : {correct/total * 100}%')
