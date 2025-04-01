import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# 损失函数对应的导数
def derivative(theta):
    return 2*(theta-2.5)

# 损失函数 theta 模型参数
def loss(theta):
    try:
       return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')


#
def gradient_descent(epllo = 10,init_theta=0,eta=0.01 , epsilon=1e-8 ):
    theta = init_theta
    history_theta = [theta]
    while epllo>0:
        epllo=epllo-1
        # 梯度
        gradient = derivative(theta)
        # 记录 上次 theta
        last_theta = theta
        # 更新参数
        theta = theta - eta * gradient
        # 计算损失
        loss_v = loss(theta)
        # 记录theta
        history_theta.append(theta)
        if(abs(loss_v - loss(last_theta)) < epsilon):
            break
    return history_theta

theta_history = gradient_descent(10,0,0.9,1e-7)
plt.plot(theta_history,loss(np.array(theta_history)),color='g',marker = '.')
print(theta_history[-1])
print(len(theta_history))
# plt.show()

# 生成样本
x,y = make_classification(n_samples=100,n_features=20)
print(x)
print(y)
print(x.shape)
print(y.shape)


x_train , x_test , y_train, y_test = train_test_split(x,y,test_size=0.50)
print(x_train)

# 权重
theta = np.random.randn(1,20) # shape (1,10)
bias = 0 #偏导

#超参数
lr = 0.01
epochs = 3000

#  点积运算
# theta =[w1,w2,w3]
# x = [[x1,x2,x3],
#      [x4,x5,x6],
#      [x7,x8,x9],]
#
# x.T =[[x1,x4,x7],
#       [x2,x5,x8],
#       [x3,x6,x9],]
#
# y 1 = w1 * x1 + w2 * x2 + w3 * x3
# y 2 = w1 * x4 + w2 * x5 + w3 * x6
# y 3 = w1 * x7 + w2 * x8 + w3 * x9
#[[y1,y2,y3]]
#模型计算函数
def forward (x,theta,bias):
    #线性运算
    z=np.dot(theta,x.T)+bias
    #sugmoid
    y_hat = 1 / (1 + np.exp(-z)) # 0，1 概率
    return y_hat

# 同 cost function
def loss(y,y_hat):
    e = 1e-8
    return -y * np.log(y_hat+e) - (1-y) * np.log(1-y_hat+e)

# 梯度计算
def calc_gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_theta = np.dot((y_hat) , x) / m
    delta_bias =  np.mean(y_hat-y)
    return delta_theta, delta_bias

# 模型训练

for i in range(epochs):
    #前向计算
    y_hat = forward(x_train,theta,bias)
    #计算损失
    loss_V = loss(y_train,y_hat)
    # 梯度计算
    delta_theta, delta_bias = calc_gradient(x_train,y_train,y_hat)
    # 更新
    theta = theta - lr * delta_theta
    bias = bias -  lr* delta_bias

    if i % 100 == 0 :
        #求准确率
        acc = np.mean(np.round(y_hat )== y_train)
        print(f"epoch:{i} , loss:{np.mean(loss(y_test, y_hat))} ,acc : {acc}")

def predict_function_usage():
    idx = np.random.randint(len(x_test)) #
    x = x_test[idx]
    y=  y_test[idx]
    predict = np.round(forward(x,theta,bias))
    print('pre = ',predict,',true=',y)
    return
predict_function_usage()


# from sklearn.datasets import load_iris
#
# x1,y1 = load_iris(return_X_y=True)
# # 前100 [:100]
# print('-----------------')
# print(x1)
# print(y1)
# print('x1.shape = ',x1.shape)
# print('x2.shape = ',y1.shape)