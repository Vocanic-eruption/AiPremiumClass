import numpy as np
from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.model_selection import train_test_split

X,y = load_iris(return_X_y=True) # 加载鸢尾花数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 参数初始化
theta = np.random.randn(1,4)
print('theta:',theta)
bias = 1
lr = 1e-1 
epochs = 10000
epsilon = 1e-8

# 向前传播
def forward(X, theta, bias):
    z = np.dot(theta, X.T) + bias  # 计算线性部分
    y_hat = 1 / (1 + np.exp(-z))  # 计算激活函数sigmoid
    return y_hat

# 损失函数
def loss(y,y_hat):
    epsilon = 1e-15  # 添加一个极小值
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # 将 y_hat 限制在 [epsilon, 1 - epsilon] 范围内
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

# 反向传播，优化器，梯度下降法
def backward(X, y, y_hat, theta, bias, lr):
    m = X.shape[0]
    dz = y_hat - y
    dtheta = np.dot(dz, X) / m
    dbias = np.sum(dz) / m
    theta -= lr * dtheta
    bias -= lr * dbias
    return theta, bias

# 训练
if __name__ == "__main__":
    for i in range(epochs):
        y_hat = forward(X_train, theta, bias)
        loss_val = loss(y_hat, y_train)
        if(abs(loss_val) < epsilon):
            break
        theta, bias = backward(X_train, y_train, y_hat ,theta, bias ,lr)
        if i % 1000 == 0:
            acc = np.mean(y_hat == y_train)
            print('Epoch: %d, loss: %.20f, accuracy: %.20f' % (i, loss_val, acc))


