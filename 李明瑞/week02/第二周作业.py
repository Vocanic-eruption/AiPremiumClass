# %%
#线性回归
# y = β0 + β1x + ϵ 
#因变量 = 截距bias + 斜率θ*自变量 + 误差项

#逻辑回归
#将线性回归模型映射为概率的模型
#把y^实数空间的输出[-∞，+∞]映射到取值为[0,1]区间 把模型输出值转换为概率值
#sigmoid函数
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# %%
#最大似然估计
#找到一个最符合当前观测数据的概率分布
#损失函数是机器学习中用于衡量模型预测值与真实值之间差异的函数
#对数似然函数 神经网络优化中最常用的损失函数
#伯努利分布
# P(X = k) = θ^k (1 - θ)^(1 - k)  k = 0, 1
# 似然函数 L(θ) = Σ(X = k) log(θ^k (1 - θ)^(1 - k))

#梯度下降法  最小化一个损失函数
#xn+1 = xn - η * δf(x, y)/δx

# %%
#梯度与学习率
#权重参数  模型中⽤于衡量输⼊特征重要程度的数值参数
#学习率  控制模型更新权重参数的速度和方向的超参数
#超参数  指模型训练过程中的不变参数，如学习率、权重参数、优化器、批次大小等

# %%
import matplotlib.pyplot as plt

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)

# %%
#梯度下降
import numpy as np
import matplotlib.pyplot as plt

# %%
plot_x = np.linspace(-1, 6, 150)  #从-1到6，分成150个点
print(plot_x)

# %%
#假设y = x^2 - 1
plot_y = (plot_x - 2.5) ** 2 - 1
plt.plot(plot_x, plot_y)    
plt.show()

# %%
#梯度下降法
#y = θ^T * x + bias

# %%
#定义损失函数对应的导数
def derivate(theta):
    return 2 * (theta - 2.5)

#定义损失函数
def loss(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')  #防止溢出  返回极大值

# %%
def gradient_descent(init_theta, eta, epsilon = 1e-8):
    #收集theta的值
    theta_history = [init_theta]
    theta = init_theta
    #记录循环次数
    iter_i = 0
    
    while iter_i < 1e4:  #重复10000次
        gradient = derivate(theta)  #参数值对应的梯度
        
        #记录上一次theta的值
        last_theta = theta
        
        theta = theta - eta * gradient  #更新参数值
        
        #记录theta
        theta_history.append(theta)
        
        #损失函数值
        loss_v = loss(theta)

        #中断循环的条件  (连续两次对应的损失函数的差值小于一个阈值)
        if abs(loss_v - loss(last_theta)) < epsilon:
            break

    return theta_history

# %%
def plot_theta_history(theta_history):
    plt.plot(plot_x, plot_y)
    plt.plot(theta_history, loss(np.array(theta_history)), color = 'r', marker = '+')
    plt.show()

# %%
theta = 0.0  #参数的初始值
eta = 0.01  #学习率
epsilon = 1e-8  #精度

# %%
theta_history = gradient_descent(theta, eta)
#训练次数
len(theta_history) - 1

# %%
#绘制损失函数变化图
plot_theta_history(theta_history)
print(theta_history[-1])

# %%
theta_history = gradient_descent(0, 0.1)
plot_theta_history(theta_history)
print(theta_history[-1])

# %%
theta_history = gradient_descent(0, 0.8)
plot_theta_history(theta_history)
print(theta_history[-1])

# %%
theta_history = gradient_descent(0, 0.2)
plot_theta_history(theta_history)
print(theta_history[-1])

# %%
theta_history = gradient_descent(6, 0.2)
plot_theta_history(theta_history)
print(theta_history[-1])

# %%
len(theta_history) - 1

# %%
#逻辑回归模型构件及训练流程
#数据准备 参数初始化  生成一组指定类别n个样本集
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %%
#1.生成训练数据
X, y = make_classification(n_samples=150, n_features=10)  #shape (150, 10)
print(X)
print(y)
print(X.shape)
print(y.shape)

#数据拆分
#局部样本训练模型（过拟合模型）
#新样本数据模型表现不好（泛化能力差）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #20%作为测试集 80%作为训练集
print(X_train.shape)
print(X_test.shape)

# %%
#权重参数
theta = np.random.randn(1,10)  #shape(1,10)  生成服从标准正太分布的随机数
bias = 0  #偏置项
#超参数
lr = 0.1  #学习率
epochs = 3000  #训练次数

# %%
#假设X是shape(3, 3)
#[x1, x2, x3]
#[x4, x5, x6]
#[x7, x8, x9]
#X.T shape(3, 3)
#[x1, x4, x7]
#[x2, x5, x8]
#[x3, x6, x9]
#假设theta模型参数shape(1, 3)
#[w1, w2, w3]
#theta * X.T
#y1 = w1 * x1 + w2 * x2 + w3 * x3
#y2 = w1 * x4 + w2 * x5 + w3 * x6
#y3 = w1 * x7 + w2 * x8 + w3 * x9

#2.模型计算函数
def forward(x, theta, bias):
    #线性运算
    z = np.dot(theta, x.T) + bias  #shape(105, 10)
    #sigmoid函数  激活函数  概率转换
    y_hat = 1 / (1 + np.exp(-z))  #shape(105, 10)
    return y_hat
#3.计算损失函数
def loss(y, y_hat):  #代价函数  cost function  
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)
#4.计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]  #样本数量
    delta_theta = np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias
#5.训练模型
for i in range(epochs):
    #前向计算
    y_hat = forward(X_train, theta, bias)
    #计算损失
    loss_val = loss(y_train, y_hat)
    #计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    #更新参数
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias
    
    if i % 100 == 0:
        #计算准确率
        acc = np.mean(np.round(y_hat) == y_train)  #四舍五入
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc:{acc}")

# %%
#模型推理
idx = np.random.randint(len(X_test)) #随机选择一个测试样本索引  随机整数
x = X_test[idx]
y = y_test[idx]

predict = np.round(forward(x, theta, bias))
print(f"y: {y}, predict: {predict}")

# %%
from sklearn.datasets import load_iris
X,y = load_iris(return_X_y=True)  #150个数据集 4个特征
X.shape

# %%
X[0]  #花瓣长度，花瓣宽度，花瓣长度，花瓣宽度

# %%
X.shape

# %%
y[0]

# %%
y

# %%
X[:100] #取前100个数据

# %%
y[:100] #取前100个标签(0,1)

# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

theta1 = np.random.randn(1,4)

bias1 = 0  

lr1 = 0.1  

epochs1 = 3000  

# %%
idx = np.random.randint(len(X_test)) #随机选择一个测试样本索引  随机整数
x = X_test[idx]
y = y_test[idx]

predict = np.round(forward(x, theta1, bias1))
print(f"y: {y}, predict: {predict}")

# %%
def forward(x, theta1, bias1):
    #线性运算
    z = np.dot(theta1, x.T) + bias1  #shape(105, 10)
    #sigmoid函数  激活函数  概率转换
    y_hat = 1 / (1 + np.exp(-z))  #shape(105, 10)
    return y_hat
#3.计算损失函数
def loss(y, y_hat):  #代价函数  cost function  
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)
#4.计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]  #样本数量
    delta_theta = np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias
#5.训练模型
for i in range(epochs1):
    #前向计算
    y_hat = forward(X_train, theta1, bias1)
    #计算损失
    loss_val = loss(y_train, y_hat)
    #计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    #更新参数
    theta1 = theta1 - lr1 * delta_theta
    bias1 = bias1 - lr1 * delta_bias
    
    if i % 100 == 0:
        #计算准确率
        acc = np.mean(np.round(y_hat) == y_train)  #四舍五入
        print(f"epoch: {i}, loss: {np.mean(loss_val)}, acc:{acc}")

# %%
#保存模型参数
np.save('theta.npy', theta1)
np.save('bias.npy', bias1)
np.save('lr.npy', lr1)
np.save('epochs.npy', epochs1)



