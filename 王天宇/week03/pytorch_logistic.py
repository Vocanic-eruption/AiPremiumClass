#使用pytorch实现逻辑回归

from sklearn.datasets import load_iris # 引入数据集
from sklearn.model_selection import train_test_split # 引入数据集划分函数 
from sklearn.datasets import make_classification
import torch
import numpy as np

# 1.1、准备样本数据
X, Y = make_classification(n_features=10)

# 1.2、将样本数据转换成张量
x_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(Y, dtype=torch.float)
print(x_tensor.shape) # torch.Size([100, 10])
print(y_tensor.shape) # torch.Size([100])

# 1.3、数据集划分
X_train, X_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.3)

# 1.4、初始化参数
weights = torch.randn(1, 10, requires_grad=True)  # 权重
bias = torch.randn(1, requires_grad=True)  # 偏置
learning_rate = 1e-3 # 学习率
iterations = 20000 # 训练次数

# 2、构建模型
for i in range(iterations):
    # 2.1、前向传播
    r = torch.nn.functional.linear(x_tensor, weights, bias)  
    r = torch.sigmoid(r)
    # 2.2、计算损失
    loss = torch.nn.functional.binary_cross_entropy(r.squeeze(1), y_tensor, reduction='mean')
    #2.3、计算梯度
    loss.backward()
    with torch.autograd.no_grad(): # 禁用梯度计算
        #2.4、更新参数
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
        weights.grad.zero_() # 清空梯度
        bias.grad.zero_() 
    if (i+1) % 1000 == 0:  # 每1000次迭代打印一次损失
        print(f"Iteration {i+1}/{iterations}, Loss: {loss.item()}")
        
# 预测函数
def predict(x):
    r = torch.nn.functional.linear(x, weights, bias)
    r = torch.sigmoid(r)
    #tensor转numpy
    r = r.detach().numpy()
    return r

idx = np.random.randint(len(X_test))
x_res = X_test[idx]
y_act = y_test[idx]
y_pred = predict(x_res)
print(f"y: {y_act}, predict: {y_pred}")

