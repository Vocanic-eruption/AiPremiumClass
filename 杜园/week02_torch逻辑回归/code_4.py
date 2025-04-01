"""
    逻辑回归模型(AI生成的)
        逻辑回归是一种分类算法，用于预测二元或多元分类问题。
        逻辑回归的输出是一个概率值，表示样本属于某个类别的概率。
        逻辑回归的损失函数是对数损失函数，用于衡量模型预测结果与真实结果之间的差距。
        逻辑回归的优化算法是梯度下降算法，用于最小化损失函数。
    理解
        目标：求n个样本同时发生概率最大值，即损失函数损失函数最小的时候求取θ向量最优解
        步骤：
            1、初始化参数(训练数据集、学习率、迭代次数、初始化特征向量参数和截距)
            2、定义损失函数（原预测函数带入sigmoid函数求事件发生概率最大值（取反即求取损失函数最小值））
            3、定义梯度下降算法（利用损失函数的偏导数公式计算）
            4、模型训练（循环迭代次数，每次迭代都需要计算预测值、损失函数值、梯度值，然后更新参数θ和bias）
            5、模型参数保存到文件中
"""

# scikit-learn用于生成分类问题的数据集
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 生成150个样本，每个样本有10个特征
X, y = make_classification(n_samples=150, n_features=10)
# X每个样本的特征值
print(X)
# y每个样本对应的标签 
print(y)
print(X.shape)

# 数据拆分
# 使用此步原因：
#   局部样本训练模型（过拟合模型）测试预测不好
#   新样本数据模型表现不好（泛化能力差）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 1、初始化参数
# θ 特征参数向量
theta = np.random.randn(1, 10)
# bias 截距
bias = 0
# α 学习率
eta = 0.001
# 迭代次数
epochs = 10000

# 2、定义损失函数(需要传入预测值和真实值)
def loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 3、定义模型计算函数(获取预测值)
def model(X, p_theta, p_bias):
    z = np.dot(p_theta, X.T) + p_bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat


# 4、计算梯度（利用损失函数的偏导数公式计算）
def cal_gradient(X, y, y_hat):
    # 特征数量
    m = X.shape[-1]
    # theta梯度计算
    delta_theta = np.dot((y_hat - y), X) / m
    # bias梯度计算
    delta_bias = np.sum(y_hat - y) / m
    return delta_theta, delta_bias
    
# 5、模型训练
def train():
    now_theta = theta
    now_bias = bias
    for i in range(epochs):
        # 计算预测值
        y_hat = model(X_train, now_theta, now_bias)
        # ① 计算损失函数值
        loss_val = loss(y_train, y_hat)
        # ② 计算梯度
        delta_theta, delta_bias = cal_gradient(X_train, y_train, y_hat)
        # 更新参数θ和bias
        now_theta = now_theta - eta * delta_theta 
        now_bias = now_bias - eta * delta_bias
        
        if i % 100 == 0:
            # 计算准确率-> 预测值与真实值比较 针对预测值四舍五入再比较（四舍五入后都是0或1），
            # 如果预测值与真实值相等则为True，否则为False 对应0或1
            # 计算准确率：预测值与真实值相等的数量/总数量 -> 准确率
            acc = np.mean(np.round(y_hat) == y_train)  # [False,True,...,False] -> [0,1,...,0]
            print(f"循环次数-epoch: {i}, 损失值-loss: {np.mean(loss_val)}, 准确率-acc: {acc}")
    return now_theta, now_bias

# 模型训练
theta, bias = train()
print("训练完成后的theta:", theta)
print("训练完成后的bias:", bias)

# 6、模型参数保存到文件中
np.save("theta.npy", theta)
np.save("bias.npy", bias)
