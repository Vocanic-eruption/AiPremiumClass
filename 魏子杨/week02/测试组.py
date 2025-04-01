import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 测试模型
#前向计算
def forward(x, theta, bias):
    z = np.dot(theta,x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

def predict(x):
    theta = np.load('theta1.npy')
    bias = np.load('bias1.npy')
    pred = forward(x,theta,bias)[0]
    if pred > 0.5:
        return 1
    else: 
        return 0
#获取数据
X,y=load_iris(return_X_y=True)
#只要前一百
test_X,test_y=X[:100],y[:100]

idx = np.random.randint(len(test_X))
x = test_X[idx]
y = test_y[idx]


pred = predict(x)
print(f'预测值：{pred} 真实值：{y}')