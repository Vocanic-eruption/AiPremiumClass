import numpy as np
import matplotlib.pyplot as plt

# 线性回归模型
# 梯度下降法演示

# 计算函数
def fun(x):
    try:
        return (x - 2.5) ** 2 - 1
    except:
        return float('inf')

# 计算导数
def derivate(theta):
    return 2 * (theta - 2.5)

# 计算损失函数
def loss(theta):
    return fun(theta)

def gradient_descent(initial_theta, eta, epsilon=1e-8, max_iter=1e4):
    # 记录theta值
    theta_history = []

    # 初始循环次数
    iter_count = 0

    theta = initial_theta
    theta_history.append(theta)

    while iter_count < max_iter:
        # 计算当前theta值对应的梯度
        gradient = derivate(theta)

        # 记录当前的theta值
        last_theta = theta

        # 更新theta值，向梯度的反方向移动，步长使用学习率eta来控制
        theta = theta - eta * gradient
        theta_history.append(theta)

        # 中断循环条件
        # 连续两次损失函数值的差值小于epsilon，则认为收敛到了最小值
        if (abs(loss(theta) - loss(last_theta))) < epsilon:
            break

        # 增加循环次数
        iter_count += 1

    return theta, theta_history

# 全局变量，用于记录绘图的序号
figure_num = 0

# 绘制函数图像
def plot_fun(title='Gradient Descent'):

    # 创建一个新窗口
    global figure_num
    figure_num += 1
    plt.figure(figure_num)
    plt.title('Gradient Descent')

    plot_x = np.linspace(-1, 6, 150)
    plot_y = fun(plot_x)
    plt.plot(plot_x, plot_y)
    pass

# 绘制theta值的变化
def plot_theta_history(theta_history, title='Theta History'):
    
    # 创建一个新窗口
    global figure_num
    figure_num += 1
    plt.figure(figure_num)
    plt.title('Theta History')

    plot_x = np.linspace(-1, 6, 150)
    plot_y = fun(plot_x)
    plt.plot(plot_x, plot_y)
    plt.plot(np.array(theta_history), fun(np.array(theta_history)), c='r', marker='+')
    pass

# 显示所有图像
def plot_show_all():
    plt.show()
    pass

if __name__ == '__main__':
    # 初始化参数值
    theta = 0.0 # 初始值
    # eta = 0.1 # 学习率
    # eta = 0.3
    # eta = 0.7
    eta = 0.9
    epsilon = 1e-8 # 精度


    # 绘画函数图像
    plot_fun()

    # 训练模型
    theta, theta_history = gradient_descent(theta, eta, epsilon)

    # 输出结果
    print(theta)
    print(loss(theta))

    # 绘制theta值的变化
    plot_theta_history(theta_history)

    # 显示图像
    plot_show_all()
    pass
