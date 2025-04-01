#第一部分
import numpy as np
import matplotlib.pyplot as plt

#创建150个数值的等差数列，代表模型的参数取值
plot_x=np.linspace(-1,6,150)
print(plot_x)

# 通过⼆次⽅程来模拟⼀个损失函数的计算,plot_y值就是图形上弧线所对应的点
plot_y=(plot_x-2.5)**2-1

plt.plot(plot_x,plot_y)
plt.show()

#定义函数求theta对应损失函数的导数
def derivative(theta):
    return 2*(theta-2.5)

#定义函数计算theta对应的损失函数
def loss(theta):
    return(theta-2.5)**2-1

#计算梯度值的过程
# # 以0作为theta的初始点
theta = 0.             #梯度
eta = 0.1              #学习率
epsilon = 1e-8         #极小值epsilon 表⽰我们的需要theta达到的最⼩值⽬标
while True:
# 计算当前theta对应点的梯度(导数)
 gradient = derivative(theta)
# 更新theta前,先积累下上⼀次的theta值，记录下来
 last_theta = theta
# 更新theta,向导数的负⽅向移动⼀步，步⻓使⽤eta(学习率)来控制
 theta = theta - eta * gradient
# 理论上theta最⼩值应当为0是最佳。但实际情况下，theta很难达到刚好等于0的条件
#所以我们可以设置⼀个最⼩值epsilon来表⽰我们的需要theta达到的最⼩值⽬标
# 判断theta每次更新后,和上⼀个theta的差值是否已满⾜⼩于epsilon(最⼩值)的条件
# 满⾜的话就终⽌运算
 if(abs(loss(theta) - loss(last_theta)) < epsilon):
    break
print(theta)
print(loss(theta))

#探究学习率对梯度的影响
theta = 0.0          #梯度
eta = 0.1            #学习率
epsilon = 1e-8       #极小值
# 添加⼀个记录每⼀步theta变更的list
theta_history = [theta]
while True:
   gradient = derivative(theta)
   last_theta = theta
   theta = theta - eta * gradient
# 更新theta后记录它们的值
   theta_history.append(theta)
   if(abs(loss(theta) - loss(last_theta)) < epsilon):
      break

plt.plot(plot_x,loss(plot_x))
plt.plot(np.array(theta_history),loss(np.array(theta_history)),color="r",marker='+')
plt.show()
len(theta_history)

def derivative(theta):
    return 2*(theta-2.5)
def loss(theta):
    return(theta-2.5)**2-1

#封装为两个函数
theta_history = []
def gradient_descent(initial_theta=0, eta=0.1, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)

theta_history = [theta]

while True:
    gradient = derivative(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
 
    if(abs(loss(theta) - loss(last_theta)) < epsilon):
        break
 
def plot_theta_history():
    plt.plot(plot_x, loss(plot_x))
    plt.plot(np.array(theta_history),loss(np.array(theta_history)), color="r", marker='+')
    plt.show()
print(theta_history)

eta = 0.01
theta_history = []
gradient_descent(0, eta)
plot_theta_history()
len(theta_history)

eta = 0.0001
theta_history = []
gradient_descent(0, eta)
plot_theta_history()
len(theta_history)

eta = 0.8
theta_history = []
gradient_descent(0, eta)
len(theta_history)

#通过异常来改造损失值计算⽅法
def loss(theta):
 try:
    return (theta-2.5)**2 - 1.
 except:
    return float('inf') # 计算溢出时，直接返回⼀个float的最⼤值

#给梯度更新⽅法添加⼀个最⼤循环次数
def gradient_descent(initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
 
    theta = initial_theta
    i_iter = 0    # 初始循环次数
    theta_history.append(initial_theta)
    while i_iter < n_iters: # ⼩于最⼤循环次数
        gradient = derivative(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
 
        if(abs(loss(theta) - loss(last_theta)) < epsilon):
            break
    i_iter += 1 # 循环次数+1

eta = 1.1
theta_history = []
gradient_descent(0, eta)
len(theta_history)

#第二部分
from sklearn.datasets import make_classification
train_X,test_X, train_y,test_y = train_test_split(X,y, test_size=0.2)

#两种参数：权重参数与超参数
# weight parameters
theta = np.random.randn(1, 10)
bias = 0
# hyper parameters
lr = 1e-3
epoch = 5000

#前向运算过程
def forward(x, theta, bias):
    # linear
    z = np.dot(theta,x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

#损失计算
def loss_function(y, y_hat):
    e = 1e-8 # 防⽌y_hat计算值为0，添加的极⼩值epsilon
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

#计算梯度
def calc_gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat-y,x)/m
    delta_b = np.mean(y_hat-y)
    return delta_w, delta_b

#更新参数
theta -= lr * dw
bias -= lr * db

#调整学习率
for i in range(epoch):
 #正向
    y_hat = forward(train_X,theta,bias)
 #计算损失
    loss = np.mean(loss_function(train_y,y_hat))
    if i % 100 = 0:
    print('step:',i,'loss:',loss)
 #梯度下降
    dw,db = calc_gradient(train_X,train_y,y_hat)
 #更新参数
    theta -= lr * dw
    bias -= lr * db

#模型测试
idx=np.random.randint(len(test_X))
x=test_X[idx]
y=test_y[idx]
def predict(x):
    pred = forward(x,theta,bias)[0]
    if pred > 0.5:
        return 1
    else: 
        return 0
pred = predict(x)
print(f'预测值：{pred} 真实值：{y}')


