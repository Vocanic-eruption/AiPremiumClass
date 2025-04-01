import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from face_model import draw_train_hist


# 定义模型
class Olivettiface(nn.Module):
    """
    只有优化器的情况下，loss、acc变化如下：

    优化器：Adam + lr=0.001 + epoch=100
    Epoch: 100/100 | Loss: 0.0409 | Train Acc: 0.9375

    修改优化器：Adam -> SGD + momentum=0.9  + lr=0.01 + epoch=100
    Epoch: 100/100 | Loss: 0.0576 | Train Acc: 0.9250
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.activate = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        log_x = self.linear_1(x)
        act_x = self.activate(log_x)
        yp = self.linear_2(act_x)
        if y is not None:
            return self.loss_fn(yp, y)
        else:
            return yp


# 获取数据, 划分数据集
def get_data():
    data_path = os.path.dirname(__file__)
    faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
    X = faces.data
    Y = faces.target
    # 小样本多类别的数据集中，加上stratify=y，以确保类别分布均衡（不加的话可能导致训练集中有某些类别的样本，而测试集中没有该类别样本，或上述相反情况）
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=35)
    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )


# 测试准确率
def test_accuracy(model, test_data, device):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for x, y in test_data:
            x, y = x.to(device), y.to(device)  # 传输到设备
            outputs = model(x)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 取最大概率的类别
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    # 配置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 配置参数
    epochs = 20
    batch_size = 80
    lr = 0.1
    input_size = 4096
    hidden_size = 512
    output_size = 40
    # 创建模型
    model = Olivettiface(input_size, hidden_size, output_size)
    # 配置优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 获取数据
    X_train, y_train, X_test, y_test = get_data()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 配置模型计算设备
    model.to(device)
    for epoch in range(epochs):
        # 训练
        model.train()
        # 收集loss，观察变化
        watch_loss = []
        # 划分batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 梯度归零
            optim.zero_grad()
            # 前向传播
            loss = model(x, y)
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()
            # 收集loss
            watch_loss.append(loss.item())
        acc = test_accuracy(model, test_loader, device)
        # 输出loss和本轮epoch的准确率
        print(f"Epoch: {epoch + 1:2d}/{epochs} | "
              f"Loss: {np.mean(watch_loss):.4f} | "
              f"Train Acc: {acc:.4f}")


if __name__ == '__main__':
    main()

# -*- encoding:utf-8 -*-
'''
@Author: 阿布
@Date: 2025/03/18 14:26
@File: hw2_bn_olivettiface.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 定义模型
class Olivettiface(nn.Module):
    """
    epochs = 50
    lr = 0.01
    Epoch: 50/50 | Loss: 0.0005 | Train Acc: 0.9500
    优化器: SGD + momentum=0.9
    Epoch:  9/50 | Loss: 0.1144 | Train Acc: 0.9750
    Epoch: 10/50 | Loss: 0.0866 | Train Acc: 0.9750
    优化器：Adam
    Epoch:  9/50 | Loss: 0.0101 | Train Acc: 0.8750
    Epoch: 10/50 | Loss: 0.0064 | Train Acc: 0.9250
    Epoch: 11/50 | Loss: 0.0046 | Train Acc: 0.9500
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activate = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        log_x = self.linear_1(x)
        bn_x = self.bn1(log_x)
        act_x = self.activate(bn_x)
        y_pred = self.linear_2(act_x)
        if y is not None:
            return self.loss_fn(y_pred, y)
        else:
            return y_pred


# 获取数据，划分数据集
def get_data():
    data_path = os.path.dirname(__file__)
    faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
    X = faces.data
    Y = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=35)

    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )


# 测试准确率
def test_accuracy(model, test_data, device):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for x, y in test_data:
            x, y = x.to(device), y.to(device)  # 传输到设备
            outputs = model(x)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 取最大概率的类别
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    batch_size = 80
    lr = 0.01
    input_size = 4096
    hidden_size = 512
    output_size = 40

    model = Olivettiface(input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # 获取数据
    X_train, y_train, X_test, y_test = get_data()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        # 计算测试准确率
        acc = test_accuracy(model, test_loader, device)
        print(f"Epoch: {epoch + 1:2d}/{epochs} | "
              f"Loss: {np.mean(watch_loss):.4f} | "
              f"Train Acc: {acc:.4f}")
    torch.save(model.state_dict(), 'bn_olivettiface.pth')




if __name__ == '__main__':
    main()

# -*- encoding:utf-8 -*-
'''
@Author: 阿布
@Date: 2025/03/18 14:26
@File: hw2_dropout_olivettiface.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 定义模型
class Olivettiface(nn.Module):
    """
    dropout层时，loss、acc变化如下：
    增大训练轮数、降低学习率，最终acc到90以上
    lr由01降低为0.001, epoch由20增大为100, acc从0.025增大为0.96
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.dp = nn.Dropout(p=0.2)
        self.activate = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        log_x = self.linear_1(x)
        act_x = self.activate(log_x)
        dp_x = self.dp(act_x)
        y_pred = self.linear_2(dp_x)
        if y is not None:
            return self.loss_fn(y_pred, y)
        else:
            return y_pred


# 获取数据，划分数据集
def get_data():
    data_path = os.path.dirname(__file__)
    faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
    X = faces.data
    Y = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=35)
    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )


# 测试准确率
# 测试准确率
def test_accuracy(model, test_data, device):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for x, y in test_data:
            x, y = x.to(device), y.to(device)  # 传输到设备
            outputs = model(x)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 取最大概率的类别
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 80
    lr = 0.001
    input_size = 4096
    hidden_size = 512
    output_size = 40

    model = Olivettiface(input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # 获取数据
    X_train, y_train, X_test, y_test = get_data()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        # 计算测试准确率
        acc = test_accuracy(model, test_loader, device)
        print(f"Epoch: {epoch + 1:2d}/{epochs} | "
              f"Loss: {np.mean(watch_loss):.4f} | "
              f"Train Acc: {acc:.4f}")


if __name__ == '__main__':
    main()
# -*- encoding:utf-8 -*-
'''
@Author: 阿布
@Date: 2025/03/18 14:26
@File: hw2_dropout_bn_olivettiface.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 定义模型
class Olivettiface(nn.Module):
    """
    bn + dropout:
    1. lr 由 0.1 -> 0.01
    2. epoch 由 20 - > 100
    3. acc 由 0.9 -> 0.975
    Epoch: 100/100 | Loss: 0.0004 | Train Acc: 0.9750
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(p=0.3)
        self.activate = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        log_x = self.linear_1(x)
        bn_x = self.bn1(log_x)
        act_x = self.activate(bn_x)
        dp_x = self.dp(act_x)
        y_pred = self.linear_2(dp_x)
        if y is not None:
            return self.loss_fn(y_pred, y)
        return y_pred


# 获取数据，划分数据集
def get_data():
    data_path = os.path.dirname(__file__)
    faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
    X = faces.data
    Y = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=35)
    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )


# 测试准确率
def test_accuracy(model, test_data, device):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for x, y in test_data:
            x, y = x.to(device), y.to(device)  # 传输到设备
            outputs = model(x)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 取最大概率的类别
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 80
    lr = 0.01
    input_size = 4096
    hidden_size = 512
    output_size = 40

    model = Olivettiface(input_size, hidden_size, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    # 获取数据
    X_train, y_train, X_test, y_test = get_data()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        # 计算测试准确率
        acc = test_accuracy(model, test_loader, device)
        print(f"Epoch: {epoch + 1:2d}/{epochs} | "
              f"Loss: {np.mean(watch_loss):.4f} | "
              f"Train Acc: {acc:.4f}")


if __name__ == '__main__':
    main()
# -*- encoding:utf-8 -*-
'''
@Author: 阿布
@Date: 2025/03/18 14:25
@File: hw3_bn_mutil_optim_olivettiface.py
'''
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# 定义模型
class Olivettiface(nn.Module):
    """
    基于bn层，测试不同的优化器，loss、acc变化如下

    调整：
    epochs = 100
    lr = 0.01

    loss: 0.0127, acc: 0.9750
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activate = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        log_x = self.linear_1(x)
        bn_x = self.bn1(log_x)
        act_x = self.activate(bn_x)
        y_pred = self.linear_2(act_x)
        if y is not None:
            return self.loss_fn(y_pred, y)
        else:
            return y_pred


# 获取数据，划分数据集
def get_data():
    data_path = os.path.dirname(__file__)
    faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
    X = faces.data
    Y = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=35)
    return (
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )


# 测试准确率
def test_accuracy(model, test_data, device):
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for x, y in test_data:
            x, y = x.to(device), y.to(device)  # 传输到设备
            outputs = model(x)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 取最大概率的类别
            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    batch_size = 80
    lr = 0.01
    input_size = 4096
    hidden_size = 512
    output_size = 40
    optims = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }

    # 获取数据
    X_train, y_train, X_test, y_test = get_data()
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for o_name, optim in optims.items():
        model = Olivettiface(input_size, hidden_size, output_size).to(device)
        optim = optim(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            watch_loss = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                loss = model(x, y)
                loss.backward()
                optim.step()
                watch_loss.append(loss.item())
            # 计算测试准确率
            acc = test_accuracy(model, test_loader, device)
            print(f"Optim:{o_name} | "
                  f"Epoch: {epoch + 1:2d}/{epochs} | "
                  f"Loss: {np.mean(watch_loss):.4f} | "
                  f"Train Acc: {acc:.4f}")


if __name__ == '__main__':
    main()