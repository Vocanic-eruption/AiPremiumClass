#本周作业：
# 1. 使用pytorch搭建神经网络模型，实现对KMNIST数据集的训练。
# https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST
# 2. 尝试调整模型结构（变更神经元数量，增加隐藏层）来提升模型预测的准确率
# 3. 调试超参数，观察学习率和批次大小对训练的影响。
from datetime import datetime
import torch
from matplotlib.ticker import MultipleLocator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from matplotlib import pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定在 gpu上工作
# torch.set_default_device('cuda')


def check_device():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'use {device} ')
    return device
# check device
device = check_device()
torch.set_default_device(device)

def load_FashionMNIST():
    trainData = datasets.FashionMNIST(
        root='../data',
        train=True, # 训练集
        download=True,
        transform=transforms.ToTensor()  # 原始数据转换成 张量
    )
    testData = datasets.FashionMNIST(
        root='../data',
        train=False, # 测试集
        download=True,
        transform=transforms.ToTensor()
    )
    return trainData , testData

def load_KMNIST():
    transform_train = transforms.Compose([
        transforms.RandomRotation(16),
        transforms.RandomAffine(7,translate=(0.13,0.11)),
        transforms.ToTensor()
    ])

    train_data = datasets.KMNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = datasets.KMNIST(
        root='../data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    return train_data, test_data

def load_DataLoader(MNIST ,BATCH_SIZE):
    train_data, test_data = MNIST
    #使用数据加载器 批量加载
    train_data = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True,
                            generator=torch.Generator(device=device))
    test_data = DataLoader(test_data, batch_size=BATCH_SIZE ,
                           generator=torch.Generator(device=device))
    return train_data, test_data

# train,test = load_DataLoader(BATCH_SIZE=1)

def data_usage():
    train_data = datasets.KMNIST(
        root='../data',
        train=True,
        download=True,
    )
    print(len(train_data)) # 数据长度 -> 60000
    # print(train_data.data) # 元组（Tuple）中两个 元素
    data, clazz = train_data[0]
    print(data)
    print(clazz)
    labels = set([])
    #  set([clazz for  img, clazz in train_data]) 等于如下
    for data_item,clazz_item in train_data:
        labels.add(clazz_item)
    print(labels) # 判断所有 分类共计 10 种 ： {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

#参考流程
# X 输入 shape（，784）# 784个特征 （28*28）
# 隐藏层 shape（784，64） # 神经元数量 ，参数矩阵 ， 线性层、 sigmoid
# 隐藏层 shape（64，） # 偏置 bias
# 输出层 shape (64,10) # 参数矩阵
# 输出层 shape (10，) # 偏置 bias
# Y 输出 shape（，10）# 10个类别
class TorchNeuralNet(nn.Module):
    def __init__(self):
        super(TorchNeuralNet, self).__init__()
        self.flatten = nn.Flatten() # 展开所有张量为 1 维
        # Sequential 顺序模型
        self.linear_Sigmoid_Sequential = nn.Sequential(
            nn.Linear(784, 588),
            # nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(588, 256),
            # nn.GELU(),
            # nn.Linear( 512,256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),


        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_Sigmoid_Sequential(x)
        print(logits.shape)
        return logits




# def predict(data):
#     output = model(data.reshape(-1, 784))
#     print(f"预测 = {output}")


def Obs(times,EPOCHS,BATCH_SIZE,history_epoch,history_loss,history_acc):
    plt.xlabel(f'[loss: value + 0.5] SUM EPOCHS = {EPOCHS} ')
    plt.ylabel(f' BATCH_SIZE = {BATCH_SIZE},LR = {LR}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.plot(range(len(history_loss)), history_loss, label='loss', color='blue')
    plt.plot(range(len(history_acc)), history_acc, label='acc', color='red')
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.legend(loc='best')
    plt.savefig(f'[{times}]_{history_epoch[-1]}_loss_acc.png')
    plt.show()



def test_torch():
    torch.fit



if __name__ == '__main__':

    # data_usage() # 测试数据

    EPOCHS = 100
    BATCH_SIZE =1024
    # LR = 1e-3
    LR = 0.083
    train_data, test_data = load_DataLoader(load_KMNIST(),BATCH_SIZE)
    # train_data, test_data = load_DataLoader(load_FashionMNIST(), BATCH_SIZE)

    model = TorchNeuralNet().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.87)
    model.train()
    history_loss = []
    history_epoch = []
    history_acc = []

    times  = datetime.now().strftime("%H%M%S")
    for epoch in range(EPOCHS):
        for x, y  in train_data:
            x = x.to(device)
            y = y.to(device)
            #forward
            pred_y=model(x.reshape(-1,784))
            #loss
            loss = loss_function(pred_y,y) #交叉熵损失函数
            #更新参数
            model.zero_grad() # 模型梯度参数清空
            loss.backward()
            optimizer.step()


        # print(f'epoch {epoch}, loss {loss.item()}')

        # 测试
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_data:
                data = data.to(device)
                target = target.to(device)
                output = model(data.reshape(-1, 784))
                _, predicted = torch.max(output.data, 1)  # 返回每行最大
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print(f'epoch {epoch}, loss {loss.item():.6f} , Accuracy : {correct / total * 100:.6f}%')

        history_acc.append(correct/total )
        history_loss.append(loss.item()+0.5)
        history_epoch.append(epoch)
        if (epoch!= 0 ) and ( epoch % 10 == 0):
            Obs(times,EPOCHS,BATCH_SIZE,history_epoch,history_loss,history_acc)
            # history_acc = []
            # history_epoch = []
            # history_loss = []
            torch.save(model, '../data/MyModel_KMNIST.pth')

