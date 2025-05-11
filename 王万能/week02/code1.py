import numpy as np

import matplotlib.pyplot as plt

for i in range(1, 10):
    print(i / 10, np.log(i / 10))

plot_x = np.linspace(-1, 6, 150)
print(plot_x)

# 线性输出
plot_y = (plot_x - 2.5) ** 2 - 1
plt.plot(plot_x, plot_y)
plt.show()


# 计算导数
def derivate(theta):
    return 2 * (theta - 2.5)


# 计算损失函数
def loss(theta):
    try:
        return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')



# 梯度下降
def gradient_descent(init_theta, eta, epsilon=1e-8):
    theta_history = [init_theta]
    theta = init_theta
    # 记录循环次数
    iteration = 0
    while iteration < 8:
        # 参数对应的梯度
        gradient = derivate(theta)
        # 记录上一次的theta
        last_theta = theta
        # 更新参数值
        theta = theta - eta * gradient
        # 记录theta值
        theta_history.append(theta)
        # 计算损失函数
        loss_v = loss(theta)
        # 损失函数小于阈值
        if abs(loss_v - loss(last_theta)) < epsilon:
            break
        iteration += 1
    return theta_history

def plot_theta_history(theta_history):
    plt.plot(plot_x, plot_y)
    plt.plot(theta_history,loss(np.array(theta_history)),color='red',marker='+')
    plt.show()


#  参数初始值
theta = 0.0
# 学习率
eta = 0.8
# 精度
epsilon = 1e-8

theta_history = gradient_descent(theta, eta, epsilon)
# 训练次数
print(len(theta_history))
plot_theta_history(theta_history)
print(theta_history[-1])

