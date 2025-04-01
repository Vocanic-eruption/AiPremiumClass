from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 1.生成训练数据
# 生成随机分类数据集
def make_dataset_classification(num_samples=150, num_features=10, test_ratio=0.3):
    # 生成数据集
    X, Y = make_classification(num_samples, num_features)
    # 数据拆分
    # 局部样本训练模型（过拟合模型）测试预测不好
    # 新样本数据模型表现不好（泛化能力差）
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = test_ratio)
    return train_X,  train_Y, test_X, test_Y

# 生成鸢尾花数据集，数据是固定的
def load_dataset_iris(test_ratio=0.3):
    X, Y = load_iris(return_X_y=True)
    # 取前100个样本的特征和标签
    train_X, test_X, train_Y, test_Y = train_test_split(X[:100], Y[:100], test_size = test_ratio)
    return train_X,  train_Y, test_X, test_Y

# 数据集保存
def save_datasets(train_X, train_Y, test_X, test_Y, file_name="datasets.npz"):
    np.savez(file_name, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
    pass

# 训练数据集加载
def load_train_data(file_name="datasets.npz"):
    data = np.load(file_name)
    train_X = data['train_X']
    train_Y = data['train_Y']
    return train_X, train_Y

# 测试数据集加载
def load_test_data(file_name="datasets.npz"):
    data = np.load(file_name)
    test_X = data['test_X']
    test_Y = data['test_Y']
    return test_X, test_Y

# 生成权重参数
def make_weights(num_features=10):
    theta = np.random.randn(1, num_features)
    return theta

# 保存权重参数
def save_weights_bias(theta, bias, file_name="weights.npz"):
    np.savez(file_name, theta=theta, bias=bias)
    pass

# 加载权重参数
def load_weights_bias(file_name="weights.npz"):
    data = np.load(file_name)
    theta = data['theta']
    bias = data['bias']
    return theta, bias

# 2.模型计算函数
def forward(x, theta, bias):
    # 线性运算
    z = np.dot(theta, x.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 3.计算损失函数
def loss_function(y, y_hat):
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 4.计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[-1]
    delta_w = np.dot(y_hat - y, x) / m
    delta_b = np.mean(y_hat - y)
    return delta_w, delta_b

# 5.更新参数
def update_params(theta, bias, delta_w, delta_b, lr):
    theta = theta - lr * delta_w
    bias = bias - lr * delta_b
    return theta, bias

# 6.模型训练，重复2至5步骤
def train(train_X, train_y, theta, bias, lr, epochs):
    for i in range(epochs):
        # 正向传递
        y_hat = forward(train_X, theta, bias)

        # 计算损失
        loss = np.mean(loss_function(train_y, y_hat))

        # 梯度计算
        delta_w, delta_b = calc_gradient(train_X, train_y, y_hat)

        # 参数更新
        theta, bias = update_params(theta, bias, delta_w, delta_b, lr)
        pass

    return theta, bias

# 7.模型测试
def predict(x, theta, bias):
    pred = forward(x, theta, bias)[0]
    if pred > 0.5:
        return 1
    else:
        return 0
