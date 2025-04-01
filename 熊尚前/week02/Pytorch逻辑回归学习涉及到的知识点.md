# Pytorch逻辑回归

线性回归：用一条直线来拟合自变量和因变量之间的关系，y=w*x+b

逻辑回归=线性回归+sigmoid函数。转换为分类问题。将线性回归的输出值，作为sigmoid函数的输入值，sigmoid函数输出0到1之间的数。线性回归的输出值（sigmoid函数的输入值）越小，sigmoid函数的输出值越趋近于0；线性回归的输出值（sigmoid函数的输入值）为0，sigmoid函数的输出值为0.5；线性回归的输出值越大，sigmoid函数的输出值越接近1。

# 标准差和方差

标准差和方差都是统计学中衡量数据离散程度（即数据分布的分散情况）的指标。

方差：1.计算一组数据的平均值。2.计算每个值与平均值的差的平方。3.第二部结果求和后除以样本数据数量。就是方差。

标准差：方差开根号就是标准差。

# 似然函数

似然函数（Likelihood Function）是统计学中用于描述已知观测数据与未知参数之间关系的函数，核心思想是：给定观测数据，参数取何值时，该数据出现的可能性最大？

要明确区分概率和似然的概念。概率是给定参数后，数据发生的可能性；似然是给定数据后，参数的可能性。

**似然函数**是 “给定数据反推参数可能性” 的工具。
**最大似然估计**是寻找使数据出现概率最大的参数值。
**与概率的区别**：概率是 “参数→数据”，似然是 “数据→参数”。

# 损失函数

损失函数是机器学习和深度学习中用于衡量模型预测结果与真实值之间差异的数学函数。它的核心作用是指导模型优化，帮助模型通过调整参数逐步减少预测误差。

【【线性回归、代价函数、损失函数】动画讲解】 https://www.bilibili.com/video/BV1RL411T7mT/?share_source=copy_web&vd_source=e792683699a0e6e0ea772adc799ae07d

# 梯度下降

是一种常用的优化算法，用于最小化目标函数。梯度下降法广泛应用于各种机器学习和深度学习算法中，用于求解模型的参数，以最小化损失函数或最大化目标函数。例如，在逻辑回归、线性回归、神经网络等模型的训练过程中，都可以使用梯度下降法来优化模型参数，使得模型能够更好地拟合训练数据，提高模型的预测性能。

# 模型训练

模型训练有如下步骤：

1. 生成训练数据

2. 数据拆分成训练数据和测试数据。

3. 权重参数

   **举例**在一个简单的线性回归模型 $y=w_1x_1 + w_2x_2 + b$中，($w_1$)和\($w_2$\)就是权重参数，分别表示特征\($x_1$\)和\($x_2$\)对输出y的影响程度。模型在训练过程中会根据输入的样本数据($x_1$, $x_2$)和对应的真实标签y，来调整\($w_1$\)和\($w_2$\)的值，以使得预测值尽可能接近真实值。

4. 超参数

   超参数是在模型训练之前需要手动设置的参数，它们**不是由模型在训练过程中直接学习得到的**，而是用于控制模型的训练过程和结构。如：学习率，训练次数

5. 模型运算

6. 损失函数

   损失函数怎么来的

7. 梯度计算

8. 模型训练

9. 模型推理

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 1.生成训练数据和测试数据
x,y = make_classification(n_features=10)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=123)
# 2.定义权重参数，theta和bias 
theta = np.random.randn(1,10) # 有10个特征
bias = 0 # 截距 能够让模型在输入特征全为 0 的情况下，也能有一个非零的输出，它起到了调整模型预测基准的作用。
# 3.定义超参数,此参数不是由模型学习得到的，而是认为定义的，用于控制训练过程和结构
lr = 1e-3 # 学习率
epochs = 1000000 # 模型训练次数
# 4.模型运算,前向运算，计算出预测值
def forward(x,theta,bias):
    z = np.dot(theta,x.T)+bias # 线性回归
    y_hat = 1/(1+np.exp(-z)) # 线性回归的输出作为sigmoid函数的输入，转换为0-1之间的概率值
    return y_hat
# 5.定义损失函数，计算损失值
def loss(y,y_hat):
    e = 1e-8 # 防止出现log0的情况
    return -y*np.log(y_hat + e)-(1-y)*np.log(1-y_hat+e)
# 6.计算梯度
def calc_gradient(x,y,y_hat):
    m = x.shape[-1] # 样本数量
    delta_w = np.dot(y_hat-y,x)/m
    delta_b = np.mean(y_hat-y)
    return delta_w,delta_b
# 7.模型训练，更新参数
flag = True
i = 0
while flag:
    # 前向运算
    y_hat = forward(train_x,theta,bias)
    # 计算损失值
    l = np.mean(loss(train_y,y_hat))
    # 观测损失值
    if i % 100 == 0:
        print(f'epoch:{i},loss:{l}')
    # 计算梯度
    delta_w,delta_b = calc_gradient(train_x,train_y,y_hat)
    # 更新参数
    theta = theta - lr*delta_w # 跟新theta
    bias = bias - lr*delta_b # 跟新bias
    i += 1
    # 训练结束条件
    if l < 1e-5 or i > epochs:
        flag = False
        print(f'训练结束，epoch:{i},loss:{l}')
# 8.模型推理
def predict(x):
    y_hat = forward(x,theta,bias)[0]
    return np.round(y_hat)
# 9.计算准确率
count = 0
for i in range(len(test_x)):
    x = test_x[i]
    y = test_y[i]
    pred = predict(x)
    if pred == y:
        count += 1
print(f'准确率:{count/len(test_x)}')

```



