from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris
import torch

# 1、生成训练数据
# x, y = make_classification(n_samples=150, n_features=10)
x, y = load_iris(return_X_y=True)
x = x[:100]
y = y[:100]
# print(x,y)
# 数据拆分
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
# 权重参数
theta = np.random.rand(1, 4)
print(theta)
bias = 0
# 学习率
lr = 0.02
# 训练次数
epochs = 3000


# 2、模型计算函数
def forward(x, theta, bias):
    # 线性计算
    z = np.dot(theta, x.T) + bias
    # sigmoid函数
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


# 3、计算损失函数
def loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)


# 4、计算梯度
def calc_gradient(x, y, y_hat):
    # 计算梯度
    m = x.shape[-1]
    # theta梯度计算
    delta_theta = np.dot((y_hat - y), x) / m
    # bias梯度计算
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias


# 模型训练
# for i in range(epochs):
#     # 向前计算
#     y_hat = forward(train_x, theta, bias)
#     # 计算损失
#     loss_val = loss(train_y, y_hat)
#     # 计算梯度
#     delta_theta, delta_bias = calc_gradient(train_x, train_y, y_hat)
#     # 更新参数
#     theta = theta - lr * delta_theta
#     bias = bias - lr * delta_bias
#
#     if i % 100 == 0:
#         acc = np.mean((np.round(y_hat) == train_y))
#         print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc: {acc}")

idx = np.random.randint(len(test_x))
x = test_x[idx]
y = test_y[idx]
print("----------------")
print(theta)
print(bias)
print("------------------")
torch.save({'theta':theta,'bias':bias}, "model.pth")

model_param = torch.load("model.pth")
theta = model_param['theta']
bias = model_param['bias']
predict = np.round(forward(x, theta, bias))
print(f"y: {y}, predict: {predict}")
print(theta)
print(bias)

