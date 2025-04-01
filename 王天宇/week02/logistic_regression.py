# 此文件定义一个逻辑回归模型
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=10000):
        self.learning_rate = learning_rate  # 学习率
        self.num_iterations = num_iterations  # 迭代次数
        self.weights = None  # 权重
        self.bias = None  # 偏置（初始预测值）

    # sigmoid函数（二分类使用sigmoid函数，将线性问题转化为0-1之间的概率问题）
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 损失函数
    def _loss(self, y, y_pred):
        e = 1e-8  # 添加一个极小值，防止出现除数为0的情况
        return (-y * np.log(y_pred + e) - (1 - y) * np.log(1 - y_pred + e)).mean()

    # 训练函数
    def fit(self, X, y):
        # 初始化权重和初始预测值
        print(X.shape[1])
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        # 梯度下降
        for i in range(self.num_iterations):
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            loss_value = self._loss(y, y_pred)
            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1 / X.shape[0]) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw  # 更新权重和偏置
            self.bias -= self.learning_rate * db  # 更新偏置
            if i % 100 == 0:
                # 计算准确率
                acc = np.mean(np.round(y_pred) == y)
                if acc > 0.8:
                    print(f"epoch: {i}, loss: {loss_value}, acc: {acc}")

    # 预测函数
    def predict(self, X):
        y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return y_pred
