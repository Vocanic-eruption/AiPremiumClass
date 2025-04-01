from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 1.生成训练数据
X, Y = make_classification(n_samples=150, n_features=10)

# 数据拆分
# 局部样本训练模型（过拟合模型）测试预测不好
# 新样本数据模型表现不好（泛化能力差）
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3)

# 权重参数
theta = np.random.randn(1,10) 
bias = 0
# 超参数
lr = 0.1
epochs = 3000  # 训练次数

# 2.模型计算函数
def forward(x, theta, bias):
    # 线性运算
    z = np.dot(theta, x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 3.计算损失函数
def loss_function(y, y_hat):
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 4.计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat - y, x) / m
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b

# 5.更新参数
def update_params(theta, bias, delta_w, delta_b, lr):
    theta = theta - lr * delta_w
    bias = bias - lr * delta_b
    return theta, bias

# 6.模型训练，重复2至5步骤
def train(train_X, train_y, theta, bias, lr, epochs):
    for i in range(epochs):
        # 正向传递
        y_hat = forward(train_X, theta, bias)

        # 计算损失
        loss = np.mean(loss_function(train_y, y_hat))

        # 梯度计算
        delta_w, delta_b = calc_gradient(train_X, train_y, y_hat)

        # 参数更新
        theta, bias = update_params(theta, bias, delta_w, delta_b, lr)
        pass

    return theta, bias


# 7.模型测试
def predict(x, theta, bias):
    pred = forward(x, theta, bias)[0]
    if pred > 0.5:
        return 1
    else:
        return 0

# 8.测试模型
idx = np.random.randint(len(test_X))
x = test_X[idx]
y = test_y[idx]
pred = predict(x, theta, bias)

print(f'预测值：{pred}, 真实值：{y}')
