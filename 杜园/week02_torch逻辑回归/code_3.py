"""
   目标：假设y = (x-2.5) ** 2 -1 为损失函数 -求最优拟合曲线
       一、目标求β1(θ：斜率)和β0(bias：截距)最优值；
       二、损失函数值最小，损失最少的时候，可求得最优的θ和bias
"""
import numpy as np
import matplotlib.pyplot as plt

# ① 获取损失函数
def loss(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except OverflowError:
        # float('inf') -> 正无穷大
        return float('inf') 
# ② 获取损失函数偏导数
def derivative(theta):
    return 2 * (theta - 2.5)

# 梯度下降过程中的参数列表
theta_his = []
# 封装梯度下降法函数
def gradient_descent(initial_theta, eta, n_iters=1e-4, eplison=1e-6):
    i_iter = 0
    theta = initial_theta
    theta_his.append(theta)
    
    while i_iter < n_iters:
        # 获取损失函数偏导数
        gradient = derivative(theta)
        last_theta = theta
        # 获取下一个theta
        theta = theta - eta * gradient
        theta_his.append(theta)
        # 终止条件
        if abs(loss(last_theta) - loss(theta)) < eplison:
            break
        i_iter = i_iter + 1
        
# 封装梯度下降函数
def plot_theta_his():
    plt.plot(theta_his, loss(np.array(theta_his)))
    plt.plot(np.array(theta_his), loss(np.array(theta_his)), color='r', marker='+')
    plt.show()

# 调用训练函数
# 不断调整学习率η，观察损失函数的变化
# 学习率越小，损失函数变化越慢
eta = 0.01
# eta = 0.01
# eta = 0.7
# 计算值超过计算机能容纳的最大值--> 梯度爆炸 -> 改造损失函数&梯度下降限制循环次数
# eta = 1.1
gradient_descent(0, eta)
print(theta_his)
print(len(theta_his))
plot_theta_his()
        
    