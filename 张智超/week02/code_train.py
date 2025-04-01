from code_datasets import datasets_path
import numpy as np
import os

# 创建文件夹保存训练后的参数
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
param_path = os.path.join(current_directory, 'trainparams')
if not os.path.exists(param_path):
    os.mkdir(param_path)

# 模型计算函数
def forward(x, theta, bias):
    # 线性运算
    z = np.dot(theta, x.T) + bias
    # 激活函数
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 损失函数
def loss_function(y_hat, y):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 梯度计算函数
def calc_gradient(y_hat, y, x):
    # 样本数量
    m = x.shape[0]
    # 权重参数梯度
    delta_w = np.dot(y_hat - y, x)/m
    # 偏置梯度
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b

def train(learning_rate, epochs):
    # 1.数据准备
    X_train = np.load(os.path.join(datasets_path, 'X_train.npy'))
    y_train = np.load(os.path.join(datasets_path, 'y_train.npy'))
    theta = np.random.randn(X_train.shape[-1])
    bias = 0

    for i in range(epochs):
        # 2.前向计算
        y_hat = forward(X_train, theta, bias)
        # 3.计算损失
        loss = loss_function(y_hat, y_train)
        if i % 100 == 0:
            # 准确率
            acc = np.mean(np.round(y_hat) == y_train)
            print(f'i={i},loss={np.mean(loss)},acc={acc}')
        # 计算梯度
        w, b = calc_gradient(y_hat, y_train, X_train)
        # 更新参数
        theta -= learning_rate * w
        bias -= learning_rate * b

    # 将训练的参数保存到文件中
    if not os.path.exists(param_path):
        os.mkdir(param_path)
    np.save(os.path.join(param_path, 'theta'), theta)
    np.save(os.path.join(param_path, 'bias'), bias)

if __name__ == "__main__":
    print('==进行模型训练(learning_rate: 学习率)==')
    train(learning_rate=1e-4, epochs=1000)
    