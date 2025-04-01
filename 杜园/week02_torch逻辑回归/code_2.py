"""
    目标：假设y = (x-2.5) ** 2 -1 为损失函数 -求最优拟合曲线
        一、目标求β1(θ：斜率)和β0(bias：截距)最优值；
        二、损失函数(一元二次方程：一般是原函数的导数)值最小，即损失最少的时候，可求得最优的θ和bias
        三、利用梯度下降法求能让损失函数达到最小值的参数θ和bias
        已知梯度下降公式：θj - α * (ðJ(θ) / ðθj)
        具体处理步骤：
            ① 获取损失函数
            ② 获取损失函数偏导数
            ③ 定义学习率
            ④ 定义θ模型参数
            ⑤ 定义计算逻辑-根据公式计算梯度值(下一个有利位置点)
            ⑥ 梯度值带入损失函数如果和上一个梯度值获取的损失函数值差距极小，即当前梯度值为最优解
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# ① 获取损失函数-只有一个参数θ->考虑如果有两个未知参数如何求最优值
def loss(theta):
    return (theta - 2.5) ** 2 - 1
# ② 获取损失函数偏导数
def derivative(theta):
    return 2 * (theta - 2.5)

# ③ 定义学习率
eta = 0.1
# ④ 定义θ模型参数
theta = 0
epsilon = 1e-8

# 定义一个观察theta变化值的数组
theta_his = []

while True:
    # ⑤ 定义计算逻辑-根据公式计算梯度值
    # 计算当前θ值处梯度(偏导数)
    gradient = derivative(theta)
    # 获取上次的theta值
    next_theta = theta
    # 利用梯度公式获取下一个步长的θ值
    theta = theta - eta * gradient
    theta_his.append(theta)
    # 理论上θ=0即最优解，但是实际上达不到，so 定义一个极小值epsilon
    # 如果下一个梯度值对应的损失函数和上一个梯度值对应的损失函数值差距特别小，即最优梯度值--所以此处需要利用循环
    if (abs(loss(theta) - loss(next_theta))) < epsilon:
        break

print(theta)
print(loss(theta))

# 经上面求解已得到原函数，下面绘制图像
# 1、样本数据
x_data = torch.linspace(-1, 6, 120)
# 绘制曲线图
plt.plot(x_data, loss(x_data))
plt.plot(np.array(theta_his), loss(np.array(theta_his)),color="r",marker="+")
plt.show()

