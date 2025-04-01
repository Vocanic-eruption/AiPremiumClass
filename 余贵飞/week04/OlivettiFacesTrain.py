# encoding: utf-8
# @File  : OlivettiFacesTrain.py
# @Author: GUIFEI
# @Desc : OlivettiFaces 数据集训练
# @Date  :  2025/03/20
import torch
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import OlivettilFacesModel as olivetti
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_path):
    '''
    加载数据、处理
    :param data_path: 数据加载路径
    :return: 返回DataLoader 类型 训练数据及测试数据
    '''
    # 下载数据集
    olivetti_faces = fetch_olivetti_faces(data_home = data_path)
    # 加载数据
    # 划分数据集
    data, target = olivetti_faces.data, olivetti_faces.target
    # 将数据划分为训练集和测试集， random_state=42 使用随机数种子确保代码每次运行时数据一致，
    # stratify=olivetti_faces.target 按照target 维度对数据集进行分层采样，却表40个人的数据都会被采集到
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42,
                                                                        stratify=olivetti_faces.target)
    # 将特征数据与目标值组合在一起
    train_data_set = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                   torch.tensor(train_target, dtype=torch.long))
    test_data_set = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                  torch.tensor(test_target, dtype=torch.long))
    # 加载数据集到 DataLoader
    train_loder = DataLoader(train_data_set, batch_size=64, shuffle=True)
    test_loder = DataLoader(test_data_set, batch_size=64, shuffle=True)
    return train_loder, test_loder


def train_model(model, train_loader, lr, epochs, optimizer_name="adam", l2_lambda=0.01):
    '''
    模型训练
    :param epochs: 迭代次数
    :param model: 模型
    :param train_loader: 训练数据
    :param lr: 学习率
    :param optimizer_name: 优化器名称，默认使用 Adam(Adam算法，通过计算梯度的⼀阶矩和⼆阶矩估计来调整学习率，对不同参数设置⾃适应学习
            率。)
    :param l2_lambda: L2 正则化强度，开启时刻开启权重衰减
    :return:
    '''
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 动态选择优化器（支持 L2 正则化）
    optimizer_params = {"params": model.parameters()}
    if model.use_l2:
        optimizer_params["weight_decay"] = l2_lambda  # L2 正则化强度
    if optimizer_name == "adam":
        optimizer = optim.Adam(**optimizer_params)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(**optimizer_params, lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    # 模型历史损失
    model_loss_history = []
    # 训练循环
    for epoch in range(epochs):
        # 开启模型训练模式
        model.train()
        model.to(device)
        for inputs, targets in train_loader:
            images, targets = inputs.to(device), targets.to(device)
            # 清空累计梯度
            optimizer.zero_grad()
            # 向前计算
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
        model_loss_history.append(loss.item())
    # 返回模型迭代历史损失数据，为后期绘图做准备
    return model_loss_history


def test(model, test_loader):
    '''
    测试模型
    :param model: 定义好的模型
    :param test_loader: 测试数据集
    :return:
    '''
    # 开启模型评估模式
    model.eval()
    model.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=1)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += inputs.shape[0]
        print(f'accuracy {100 * correct / total:.2f}%')


def draw_train_hist(hist_list):
    for i,hist in enumerate(hist_list):
        plt.plot(hist, label=f'Model{i}')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

if __name__ == '__main__':
    LR = 0.001
    EPOCHS = 20
    # 统一加载模型
    model1 = olivetti.OlivettiFaces1()
    model2 = olivetti.OlivettiFaces2()
    model3 = olivetti.OlivettiFaces3()
    model4 = olivetti.OlivettiFaces4()
    model5 = olivetti.OlivettiFaces5()
    model_list = [model1, model2, model3, model4, model5]
    # 加载数据
    train_loader, test_loader = load_data("../dataset/olivettiFaces")
    loss = []
    # 开始训练
    i = 0
    for model in model_list:
        # 模型训练
        loss_history = train_model(model, train_loader, LR, EPOCHS)
        # 保存每个模型的损失数据
        loss.append(loss_history)
        # 测试每个模型的表现
        test(model, test_loader)
    # 绘图
    draw_train_hist(loss)






