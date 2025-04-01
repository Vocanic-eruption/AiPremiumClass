import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#前向计算
def forward(x, theta, bias):
    z = np.dot(theta,x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat
# 损失函数
def loss_function(y, y_hat):
    e = 1e-8 
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 计算梯度
def calc_gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat-y,x)/m
    delta_b = np.mean(y_hat-y)
    return delta_w, delta_b



X,y=load_iris(return_X_y=True)
#print(X)
#print(y.shape)

train_X,test_X, train_y,test_y=train_test_split(X[:100],y[:100],test_size=0.4)

theta = np.random.randn(1, 4)

bias = 0
# 学习率
lr = 1e-4 #1e-3
# 模型训练的轮数
epoch = 5000


for i in range(epoch):
    #正向
    y_hat = forward(train_X,theta,bias)
    #计算损失
    loss = np.mean(loss_function(train_y,y_hat))
    #梯度下降
    dw,db = calc_gradient(train_X,train_y,y_hat)
    #更新参数
    theta -= lr * dw
    bias -= lr * db


# 保存模型参数
np.save('theta1.npy', theta)
np.save('bias1.npy', bias)

