import  numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite import color
from sympy.physics.vector import gradient
from sympy.stats import where

# 绘图
# 创建等差数列x,代表模型的参数取值,共150个
plot_x = np.linspace(-1., 6., 150)
# print(plot_x)

# 通过二次方程来模拟一个损失函数的计算,
# ？？？？ plot_y值就是图形上弧线所对应的点
plot_y = (plot_x-2.5)**2 - 1
plt.plot(plot_x, plot_y)
# plt.show()

#############################################################
## p=1/(1+loge(x))
# 激活函数 - 二元分类 [0-1]
def sigmoid(x):
    return 1/(1+np.exp(-x))

# P8 页
# y_hat  y_hat=h(x)
def loss_function(y, y_hat):
    e = 1e-8  # 防止y_hat计算值为0，添加的极小值epsilon
    y_hat = y_hat + e
    return - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

# 损失函数对应的导数
def derivative(theta):
    return 2*(theta-2.5)

# 损失函数 theta 模型参数
def loss(theta):
    try:
       return (theta - 2.5) ** 2 - 1
    except:
        return float('inf')

###########################################################
# for i in range(1,10):
#     print(i/10,np.log(i/10))

#梯度下降
def print_plot_usage():
    plot_x = np.linspace(-1., 6., 150)
    plot_y = (plot_x-2.5)**2 - 1
    plt.plot(plot_x, plot_y)
    return

# # 梯度计算
# theta  = 0.0 # init
# eta = 0.01 # 学习率
# epsilon = 0.1 #  精度


# history_theta = [0]
# while True:
#     gradient = derivative(theta) # 参考 对应theta 梯度
#
#     last_theta = theta
#     theta = theta - eta * gradient  # updata param
#     loss_v = loss(theta)
#     # print('参数 theta = ',theta)
#     history_theta.append(theta)
#
#     # break condition : 连续损失函数差 小于 阈值 。
#     if(abs(loss_v - loss(last_theta)) < epsilon):
#         print('break loss = ', loss(last_theta))
#         break
#
# print_plot_usage()
# plt.plot(history_theta,loss(np.array(history_theta)),color='r',marker = '.')
# # plt.show()
# print('training times = ',len(history_theta))

#
def gradient_descent(epochs = 10,init_theta=0,eta=0.01 , epsilon=1e-8 ):
    theta = init_theta
    history_theta = [theta]
    while epochs>0:
        epochs=epochs-1
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
plt.show()
print(theta_history[-1])
print(len(theta_history))



