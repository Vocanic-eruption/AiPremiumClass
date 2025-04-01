from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# 生成数据集
iris = load_iris()
X = iris.data
Y = iris.target
X = X[:100]   #(100,4)
Y = Y[:100]   #(1,100)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 从文件中加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
w = model['weights']
bias = model['bias']

# 随机选择测试集中的样本
idx = np.random.randint(len(X_test), size=5)
x = X_test[idx]
y = Y_test[idx]

# 定义前向传播函数
def forward(w, bias, x):
    z = np.dot(w, x.T) + bias
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 使用加载的模型进行预测
predict = np.round(forward(w, bias, x))
print(f"y: {y}, predict: {predict}")