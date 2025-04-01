# 数据准备环节
# 1. 加载数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
X,y = load_iris(return_X_y=True)
X = X[:80]  # 取前80个数据
y = y[:80]  # 取前80个标签(0,1)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('训练样本数：', X_train.shape)
print('测试样本数：', X_test.shape)
print('训练真实值：', y_train.shape)

# 2.参数设置
# 权重参数
theta = np.random.randn(1,4)
bias = 0
# 学习率
lr = 1e-2
# 迭代次数
epochs = 2000  

# 3. 模型计算函数(前向函数)
def forward(X, theta, bias):
    # 计算预测值
    z = np.dot(theta, X.T) + bias
    # sigmoid函数
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 4. 损失函数
def loss_function(y,y_hat):
    # 加上一个极小值,防止log(0)
    e = 1e-8
    return -y * np.log(y_hat+e)-(1-y) * np.log(1-y_hat+e)

# 5. 计算梯度
def calc_gradient(X, y, y_hat):
    # 特征值
    m = X.shape[-1]
    # 求theta的偏导数(计算theta梯度)
    delta_theta = np.dot((y_hat-y),X)/m
    # 求bias的偏导数(计算bias梯度)
    delta_bias = np.mean(y_hat-y)
    return delta_theta, delta_bias

# 6. 梯度更新
def update_gradient(theta, bias, delta_theta, delta_bias):
    theta = theta - delta_theta * lr
    bias = bias - delta_bias * lr
    return theta, bias

# 7. 模型训练
for i in range(epochs):
    # 前向传播(预测值)
    y_hat = forward(X_train,theta,bias)
    # 计算损失
    loss = loss_function(y_train,y_hat)
    # 计算梯度
    delta_theta, delta_bias = calc_gradient(X_train,y_train,y_hat)
    # 更新梯度
    theta, bias = update_gradient(theta, bias, delta_theta, delta_bias)

    # 计算准确率
    # if i % 50 == 0:
    acc = np.mean(np.round(y_hat) == y_train)  # [False,True,...,False] -> [0,1,...,0]
    # print(f"epoch: {i}, loss: {np.mean(loss)}, acc: {acc}")

# 8. 模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]


predict = np.round(forward(x, theta, bias))
print(f"模型推理:样本x:{x},真实值y: {y}, 预测值predict: {predict}")
print(f"保存已训练的模型参数, theta:{theta}, bias:{bias}")
np.savez("model_params.npz",theta=theta,bias=bias)
