import torch
import torch.nn as nn
import torch_nn_datasets


# X输入 shape(,784)
# 隐藏层 shape(784,128)  # 参数矩阵 # 784个特征 128个神经元
# 隐藏层 shape(128,)  # 偏置bias
# 输出层 shape(128,10)  # 参数矩阵 # 128个神经元 10个类别
# 输出层 shape(10,)  # 偏置bias
# Y输出 shape(,10)  # 10个类别

# 输入层
def input_level():
    train_data = torch_nn_datasets.load_data()
    return train_data


# 隐藏层
def hidden_level(train_data):
    # 线性函数
    hidden_linear = nn.Linear(in_features=784, out_features=128)
    h_l1 = hidden_linear(train_data)

    # 激活函数
    hidden_act = nn.Sigmoid()
    h_out1 = hidden_act(h_l1)
    return h_out1


# 输出层
def output_level(hidden_level_data):
    # 线性函数
    output_linear = nn.Linear(in_features=128, out_features=10)
    out_l1 = output_linear(hidden_level_data)

    # 激活函数
    output_softmax = nn.Softmax(dim=1)
    out_o1 = output_softmax(out_l1)

    return out_o1


# 构建简单模型
def train_model():
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.Sigmoid(),
        nn.Linear(128, 10),
    )
    return model


# 定义损失函数和优化器
def loss_func(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return loss_fn, optimizer


if __name__ == '__main__':
    pass
