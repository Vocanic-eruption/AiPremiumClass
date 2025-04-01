import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # 数据加载器

# 1. 定义超参数
# 2. 加载数据
# 3. 创建模型
# 4. 定义损失函数与优化器
# 5. 训练模型

"""
1. 神经元数量
表达能力：增加每一层中的神经元数量可以提高模型的学习能力和表达复杂模式的能力。更多的神经元意味着模型有更大的容量来捕捉数据中的细微差别和特征。
过拟合风险：然而，过多的神经元可能导致模型过于复杂，容易出现过拟合现象，即模型在训练集上表现很好但在未见过的数据（如验证集或测试集）上表现不佳。
计算成本：更多的神经元也意味着更高的计算需求，包括内存使用和处理时间。
2. 训练数据集的 Batch Size
收敛速度与稳定性：
较大的 batch_size 可以利用硬件加速（如GPU），加快每一步的梯度计算，从而可能加速训练过程。
小批次大小有助于引入更多的噪声进入梯度估计中，这可能会帮助模型跳出局部最小值，找到更优解，并且通常被认为能提供更好的泛化能力。
内存限制：较大的批次大小需要更多的内存来存储中间结果和梯度信息，这对于硬件资源有限的情况是一个挑战。
学习率调整：当增加批次大小时，可能需要相应地调整学习率。理论上，随着批次大小的增加，最优学习率也会增加。
3. 隐藏层数量
模型复杂度：增加隐藏层数量可以显著提升模型的复杂度，使其能够学习更加复杂的函数映射。对于非常复杂的问题，深层网络（具有多个隐藏层）可能是必要的。
过拟合与欠拟合：如果模型太简单（隐藏层太少），它可能无法充分学习数据中的模式，导致欠拟合；相反，如果模型过于复杂（隐藏层太多），则可能导致过拟合问题。
训练难度：更深的网络结构往往更难以优化，因为它们更容易遇到梯度消失/爆炸等问题。现代深度学习实践中常用的技术如残差连接（ResNet）、批归一化（Batch Normalization）等可以帮助缓解这些问题。
"""

# 定义超参数
# 学习率
LR = 1e-1
# 训练轮次
epochs = 20
# 批次256 准确率 83.69% 83.73% 批次128 准确率 85% 批次512 准确率81.62%、81.8% 批次64 86.57%
BATCH_SIZE = 64

# 创建模型
model = nn.Sequential(
    # 增加神经元个数 128个 准确率 83.72%  256个 83.12%  64个 83.8%
    nn.Linear(784, 64),
    nn.Sigmoid(),
    # # 隐藏层2 两层 85.69% 一层 83.8%
    nn.Linear(64, 64),
    nn.Sigmoid(),
    nn.Linear(64, 10),
)

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)


def train_model():
    train_data = FashionMNIST(root="F:/githubProject/AiPremiumClass/梁锐江/week3/fashion_data", train=True,
                              download=True, transform=ToTensor())
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # 训练轮次
    for epoch in range(epochs):
        for data, target in train_dl:
            # data是shape(不确定,28 X 28)的数据 而模型入参是784个特征 所以需要reshape
            # 前向运算得到预测值
            predict = model(data.reshape(-1, 784))
            # 计算损失
            loss = loss_fn(predict, target)
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()  # 计算梯度
            # 梯度更新
            optimizer.step()
        print(f"epoch:{epoch},loss:{loss}")


def predict_model():
    test_data = FashionMNIST(root="F:/githubProject/AiPremiumClass/梁锐江/week3/fashion_data", train=False,
                             download=True, transform=ToTensor())
    test_dl = DataLoader(test_data)

    acc_count = 0
    total = 0

    # 预测不进行梯度更新
    with torch.no_grad():
        for data, target in test_dl:
            predict = model(data.reshape(-1, 784))
            _, pred_clazz = torch.max(predict, 1)
            total += target.size(0)
            acc_count += (pred_clazz == target).sum().item()

    print(f"acc:{acc_count / total * 100}%")


if __name__ == '__main__':
    train_model()
    predict_model()
