import torch
import torch.nn as nn

'''
Batch Normalization
BatchNorm层在训练和推理时行为不同。
在训练过程中，BatchNorm计算每个小批次数据的均值和方差，并使用这些统计量对输入进行标准化。同时，它还会更新运行中的均值和方差估计，这些估计用于推理阶段。
在推理时，BatchNorm不再计算批次统计数据，而是使用训练期间累积的运行均值和方差来标准化输入。
因此，在将包含BatchNorm层的模型部分初始化后直接复用时会遇到问题，因为不同子网络可能需要不同的统计特性，如果直接复用可能导致不正确的标准化效果。
换句话说，BatchNorm层依赖于特定的数据分布情况，如果数据分布变化，BatchNorm的效果可能会变差。

Dropout
相比之下，Dropout是一种通过在训练过程中随机“丢弃”神经元（即将其输出设为零）来防止过拟合的技术。
在推理或测试阶段，Dropout是关闭的，即所有的神经元都参与计算，但它们的权重通常会乘以一个保持概率（dropout rate），以平衡由于训练时部分神经元被忽略造成的影响。
Dropout机制使得它可以比较容易地在不同的子网络之间复用，因为在推理时Dropout实际上没有作用——所有连接都是激活的。
这意味着无论上下文如何，只要是在推理模式下，Dropout的行为是一致的，不会受到其他因素的影响。
'''


class BNAndDP(nn.Module):
    def __init__(self):
        super(BNAndDP, self).__init__()
        self.ln1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.sg = nn.Sigmoid()
        self.dp = nn.Dropout(0.2)
        self.ln2 = nn.Linear(256, 10)
        self.bn2 = nn.BatchNorm1d(256)
        self.rl = nn.ReLU

    def forward(self, x):
        '''
        Dropout位置的影响
        靠近输入层：如果Dropout被放置得离输入层较近，那么它将对进入后续层的信息量产生较大影响。这可能导致信息损失，并且可能使学习变得更加困难，因为深层可能无法接收到足够的信息来进行有效的特征学习。
        靠近输出层：当Dropout层靠近输出层时，它主要作用于网络中更高层次的特征表示。这种方式有助于防止模型依赖于某些特定的高维特征，但是同样有可能导致信息丢失的问题，尤其是在输出层之前直接应用Dropout的情况下。
        夹在中间层之间：通常，Dropout最常被放置在中间隐藏层之间。这种做法允许在网络的不同阶段进行正则化，同时尽量减少信息损失的风险。这样做可以有效防止模型过拟合，并促进更鲁棒的特征学习。
        实际应用建议
        在实践中，推荐首先尝试将Dropout放置在全连接层之后、激活函数之前。这样的安排有助于确保激活函数能够处理未被Dropout稀疏化的输入。
        '''
        x = self.ln1(x)
        x = self.dp(x)
        x = self.sg(x)
        x = self.ln2(x)
        return x


if __name__ == '__main__':
    model = BNAndDP()

    train_data = torch.randn(100, 784)
    out = model(train_data)
    print(out.shape)
