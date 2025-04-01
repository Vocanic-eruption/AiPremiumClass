import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from sklearn.datasets import load_iris


# 计算theta的导数
def derivative(theta):
    return 2 * (theta - 2.5)


# 计算theta的损失函数
def loss(theta):
    return (theta - 2.5) ** 2 - 1


# 0作为theta的初始点
theta = 0
eta = 0.8
epsilon = 1e-8
# 记录theta 变化
history = [theta]
while True:
    # 对theta求导
    gradient = derivative(theta)
    # 更新theta前首先存储上一次的theta
    last_theta = theta
    # 更新theta，向导数的负方向移动一步，步长用eta来控制
    theta = theta - eta * gradient
    history.append(theta)  # 记录theta 的变化
    # 理论上theta最小值为0是最佳，我们设置最小epsilon，满足条件就终止
    if abs(loss(theta) - loss(last_theta)) < epsilon:
        break

# print(theta)
# print(loss(theta))
# print(history)

# plot_x = np.linspace(-1.0, 6.0, 150)
# plot_y = (plot_x - 2.5) ** 2 - 1
# plt.plot(plot_x, plot_y)
# plt.plot(np.array(history),loss(np.array(history)),color="r",marker="+")
# plt.show()
print("---------------------------------------------")
# 生成曲线图
# plot_x = np.linspace(-1.0, 6.0, 150)
# plot_y = (plot_x - 2.5) ** 2 - 1
# plt.plot(plot_x, plot_y)
# plt.show()

print("---------------------下面是逻辑回归------------------------")

# 生成观测值数据
X, y = make_classification(n_features=10)
print(X)
print(y)
# 拆封训练和测试集
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123
)
print(train_X)

# 初始化参数模型参数
theta = np.random.randn(1, 10)
bias = 0
# 学习率
lr = 1e-3
epoch = 5000  # 模型训练的循环次数


# 前向计算
def forward(x, theta, bias):
    z = np.dot(theta, x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


# 损失函数
def loss_function(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)


# 计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat - y, x) / m
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b


for i in range(epoch):
    # 正向
    y_hat = forward(train_X, theta, bias)
    # 计算损失
    loss = np.mean(loss_function(train_y, y_hat))

    if i % 100 == 0:
        print("step:", i, "loss:", loss)
    # 梯度下降
    dw, db = calc_gradient(train_X, train_y, y_hat)
    # 更新参数
    theta -= lr * dw

print("----------------------------下面是通过pytorch实现----------------------------")

# 超参数：学习率
learn_rate = 1e-6
X, y = make_classification(n_features=20)
# print(X)
# print(y)

# 创建张量
tensor_x = torch.tensor(X, dtype=torch.float)
tensor_y = torch.tensor(y, dtype=torch.float)


# 创建参数
w = torch.randn(1, 20, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for i in range(5000):
    # 向前运算
    r = torch.nn.functional.linear(tensor_x, w, b)
    r = torch.sigmoid(r)

    # 计算损失
    loss = torch.nn.functional.binary_cross_entropy(
        r.squeeze(1), tensor_y, reduction="mean"
    )
    
    # 计算梯度
    loss.backward()
    
    #参数更新
    with torch.autograd.no_grad():
        # 更新权重梯度
        w -= learn_rate * w.grad
        # 清空本次计算的梯度
        w.grad.zero_()
        # 更新偏置项梯度
        b -= learn_rate* b.grad
        b.grad.zero_()
        
    print(f'train loss:{loss.item():.4f}')

# 保存训练后的参数到文件
torch.save({'w': w, 'b': b}, 'model_params.pth')
