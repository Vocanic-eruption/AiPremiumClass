import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def forward(X, theta, bias):
    z = np.dot(theta, X.T) + bias
    # sigmoid
    y_hat = 1 / (1+np.exp(-z))
    return y_hat


def loss_function(y, y_hat):
    e = 1e-8
    # 逻辑回归损失函数
    loss = np.mean(-y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e))
    return loss

def gradient(x, y, y_hat):
    m = X.shape[-1]
    delta_theta = np.dot(y_hat - y, x) /m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias


def train(X, y, theta, bias, lr, epochs):
    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        y_hat = forward(X, theta, bias)
        loss = loss_function(y, y_hat)
        loss_history.append(loss)
        delta_theta, delta_bias = gradient(X, y, y_hat)
        theta = theta - lr * delta_theta
        bias = bias - lr * delta_bias
        # accuracy
        acc = np.mean(np.round(y_hat) == y)
        acc_history.append(acc)
        if epoch % 100 == 0:
            print('epoch: {}, loss: {:.6f}, acc: {:.6f}'.format(epoch, loss, acc))
    return loss_history, acc_history, theta, bias


if __name__ == "__main__":
    # 加载数据集
    X, y = load_iris(return_X_y=True)

    # 不同训练样本比率下的训练和预测情况
    test_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    loss_df = pd.DataFrame(columns=test_ratio_list)
    acc_df = pd.DataFrame(columns=test_ratio_list)
    pred_y_acc = []
    theta = np.random.randn(1, X.shape[-1])  # 给定初始theta
    bias = 0  # 初始bias
    lr = 1e-3
    epochs = 100
    for test_ratio in test_ratio_list:
        X_train, X_test, y_train, y_test = train_test_split(X[:100], y[:100], test_size=test_ratio)
        # model = LogisticRegression()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # scores = [i == j for i, j in zip(y_test, y_pred)]
        # print("Accuracy: %.2f%%" % (model.score(X_test, y_test) * 100))
        loss_history, acc_history, train_theta, train_bias = (
            train(X_train, y_train, theta, bias, lr, epochs))
        loss_df[test_ratio] = loss_history
        acc_df[test_ratio] = acc_history
        pred_y = forward(X_test, train_theta, train_bias)
        pred_acc = np.mean(np.round(pred_y) == y_test)
        pred_y_acc.append(pred_acc)
    loss_df.plot(title="Logistic Regression Loss")
    plt.show()
    plt.close()
    acc_df.plot(title="Logistic Regression Accuracy")
    plt.show()
    plt.close()
    for test_ratio, pred_acc, final_loss in zip(test_ratio_list, pred_y_acc, loss_df.iloc[-1]):
        print("test_ratio: {}, final_loss: {}, pred_acc: {}".format(test_ratio, final_loss, pred_acc))



