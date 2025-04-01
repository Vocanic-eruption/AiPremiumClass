from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

#生成数据集
iris = load_iris()
X = iris.data
Y = iris.target
X = X[:100]   #(100,4)
Y = Y[:100]   #(1,100)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#参数定义
learning_rate = 0.01
w = np.random.randn(1, 4)   #(1,4)
bias = np.random.randn(1)
epoch = 100

#定义前向传播
def forward(w,bias,x):
    z = np.dot(w, x.T) + bias  #(1,4) * (4,70) = (1,70)
    y_hat = 1 / (1 + np.exp(-z))  
    return y_hat

#定义损失函数
def loss(y, y_hat):
    e = 1e-8
    loss = -np.mean(y * np.log(y_hat + e) + (1 - y) * np.log(1 - y_hat + e))
    return loss

#定义梯度下降
def gradient_descent(w,bias,x,y,y_hat,learning_rate):
    dw = np.dot((y_hat - y) , x) / x.shape[-1]
    db = np.mean(y_hat - y)
    w = w - learning_rate * dw
    bias = bias - learning_rate * db
    return w,bias

#参数定义
learning_rate = 0.01
w = np.random.randn(1, 4)
bias = np.random.randn(1)



#模型训练
for i in range(epoch):
    y_hat = forward(w,bias,X_train)     #()
    loss_value = loss(Y_train,y_hat)
    w,bias = gradient_descent(w,bias,X_train,Y_train,y_hat,learning_rate)
    if i % 10 == 0:
        acc = np.mean(np.round(y_hat) == Y_train)
        print('epoch:',i,'loss:',loss_value,"acc:",acc)
    
# 模型保存
model = {'weights': w, 'bias': bias}
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
