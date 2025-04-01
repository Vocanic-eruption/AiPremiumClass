import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch


def get_linspace(start_data, end_data, count_nums):
    """创建等差数列函数"""
    plot_x = np.linspace(start_data, end_data, count_nums)
    return plot_x

def plot_eryuanyici(plot_x):
    """ 画二元一次方程图"""
    plot_y = (plot_x - 2.5)**2 - 1
    plt.plot(plot_x, plot_y)
    plt.show(block=False)  # 非阻塞显示图形
    plt.pause(10)  # 暂停10秒
    plt.close()  # 关闭图形窗口

def plot_draw(plotxs):
    """画图，短暂展示10s"""
    for plot_x in plotxs:
        # 将 plot_x 转换为 numpy 数组
        plot_x = np.array(plot_x)
        plt.plot(plot_x, loss(plot_x))
    plt.show(block=False)  # 非阻塞显示图形
    plt.pause(10)  # 暂停10秒
    plt.close()  # 关闭图形窗口

def derivative(theta):
    '''二元一次方程的导数梯度'''
    return 2*(theta - 2.5)

def loss(theta):
    """
    二元一次方程的损失函数
    :param theta: 计算方程的变量原始数据theta
    :return:  损失值
    """
    try:
        return (theta - 2.5)**2 - 1
    except:
        return float("inf") # 计算溢出时，直接返回一个float的最大值

def caculate_theta():
    """
    二元一次方程的 求低点流程
    :return:
    """
    theta = 0.
    eta = 1.1
    epsilon = 1e-8
    i_iter = 0
    max_iter = 10
    theta_history = [theta]
    while i_iter < max_iter:
        # 计算当前theta 对应点的梯度
        gradient = derivative(theta)

        # 更新theta前，先积累下上一次的theta 值
        last_theta = theta

        # 更新theta,向倒数的负方向移动一步，不常使用默认值
        theta = theta - eta * gradient
        theta_history.append(theta)
        if (abs(loss(theta) - loss(last_theta)) < epsilon):
            break
        i_iter += 1

    plot_x = get_linspace(-1, 6, 150)
    datas = [plot_x, theta_history]
    plot_draw(datas)
    print(len(theta_history))
    print(theta)
    print(loss(theta))

def forward(x, theta, bias):
    z = np.dot(theta, x.T) + bias
    y_hat = 1/(1 + np.exp(-z))
    return y_hat

def loss_function(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

def calc_gradient(x, y, y_hat):
    m = x.shape[0]
    delta_w = np.dot(y_hat - y, x)/m
    delta_b = np.mean(y_hat-y)
    return delta_w, delta_b

def predect(x, theta, bias):
    """调用模型预测结果"""
    pred = forward(x, theta, bias)
    if pred >0.5:
        return 1
    else:
        return 0

def draw_process(theta_history, bias_history):
    """画出训练的结果变化图"""
    plt.figure(figsize=(12, 6))
    for j in range(theta_history.shape[1]):
        plt.plot(theta_history[:, j], label=f'theta_{j}')
    plt.plot(bias_history, label='bias')
    plt.title('Theta values over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Theta and bias value')
    plt.legend()
    plt.show()

def save_model_result(theta, bias):
    """
    :param theta: 训练后的theta数据
    :param bias:  训练后bias 数据
    :return:
    """
    # result = np.array([theta])
    np.save("my_modelresult_theta.npy",theta)
    np.save("my_modelresult_bias.npy",bias)
    print("训练结果保存成功")

def load_model_result():
    """
    加载已经保存的训练结果
    :return:
    """
    theta = np.load("my_modelresult_theta.npy")
    bias = np.load("my_modelresult_bias.npy")
    print("训练结果加载成功")
    return theta,bias

def use_model_work():
    """
    使用训练好的数据，测试结果
    :return:
    """
    # 1.加载结果theta, bias
    theta, bias = load_model_result()

    # 2. 准备测试数据
    X,Y  = load_iris(return_X_y=True)
    # 模型数据拆分训练数据和测试数据
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=123)

    # 3.使用模型结果
    count_result = []
    for idx in range(len(test_x)):
        x = test_x[idx]
        pred = predect(x, theta, bias)
        y = test_y[idx]
        print("输入值", x, "预测值：", pred, "实际值：", y)
        if predect(test_x[idx], theta, bias) == test_y[idx]:
            count_result.append(1)
        else:
            count_result.append(0)
    print("准确率：", sum(count_result)/len(test_x))



def model_params_cacul():
    """基于逻辑回归模型训练和回归"""
    # 获取模型数据
    # X,Y  = make_classification(n_features= 10) # 使用随机数模型数据

    X,Y  = load_iris(return_X_y=True)     # 使用鸢尾花模型数据
    # 模型数据拆分训练数据和测试数据
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=123)
    # 随机数据模型用这个
    # theta = np.random.randn(1,10)
    # 燕尾花模型用这个
    theta = np.random.randn(1,4)
    bias = 0
    lr = 1e-5
    epoch = 500000
    theta_history = []
    bias_history = []
    for i in range(epoch):
        # 正向
        y_hat = forward(train_x, theta, bias)
        # 计算损失
        loss = np.mean(loss_function(train_y, y_hat))
        if i %100 ==0:
            print("step:", i, "loss:", loss)
        # 梯度下降
        dw, db = calc_gradient(train_x, train_y, y_hat)
        # 更新参数
        theta -= lr * dw
        bias -= lr * db
        bias_history.append(bias)
        theta_history.append(theta.flatten())

    # 保存训练结果
    save_model_result(theta, bias)
    theta_history =np.array(theta_history)
    # 计算准确率
    count_result = []
    for idx in range(len(test_x)):
        x = test_x[idx]
        y = test_y[idx]
        pred = predect(x, theta, bias)
        print("输入值", x, "预测值：", pred, "实际值：", y)
        if predect(test_x[idx], theta, bias) == test_y[idx]:
            count_result.append(1)
        else:
            count_result.append(0)
    print("准确率：", sum(count_result)/len(test_x))

    # 可视化 theta和 bias 的变化
    draw_process(theta_history, bias_history)


def torch_sklearn():
    """
    使用pytorch实现线性回归
    :return:
    """
    # 超参数：学习率
    learn_rate = 1e-3
    #1.1 数据准备
    x, y = make_classification(n_features=10)
    # 创建张量
    tensor_x = torch.tensor(x, dtype= float)
    tensor_y = torch.tensor(y, dtype= float)

    # 创建参数并初始化
    w = torch.randn(1,10, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    for i in range(50000):
        # 前向运算
        r = torch.nn.functional.linear(tensor_x, w, b)
        r = torch.sigmoid(r)

        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy(r.squeeze(1), tensor_y,reduction= "mean")

        # 计算梯度
        loss.backward()

        # 参数更新







if __name__ == '__main__':
    # 二元一次方程训练模型
    # caculate_theta()
    # 线性回归模型流程训练
    model_params_cacul()
    # 调用训练后结果
    # use_model_work()
