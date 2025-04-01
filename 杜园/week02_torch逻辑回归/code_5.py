import numpy as np
from sklearn.datasets import make_classification

# 1、读取训练的模型参数值
loaded_theta = np.load("theta.npy")
loaded_bias = np.load("bias.npy")

print("加载的 theta:", loaded_theta)
print("加载的 bias:", loaded_bias)

# 2、定义模型计算函数
def model(X, theta, bias):
    z = np.dot(theta, X.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 3、预测新样本
X, y = make_classification(n_samples=100, n_features=10)
y_hat = model(X, loaded_theta, loaded_bias)
print("预测结果：", y_hat)
print("真实结果：", y)
print("准确率：", np.mean(np.round(y_hat) == y))