#读取参数
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
theta=np.loadtxt("theta.txt")
bias=np.loadtxt("bias.txt")
print(theta)
print(bias)

#计算函数
def forward(X):
    z=np.dot(theta,X.T) + bias
    y_hat=1/(1+np.exp(-z))
    return y_hat

X, y = make_classification(n_features=10)
print(X)
print(y)
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

y_hat=forward(X_train)
#计算准确率
acc=(y_hat>0.5)==y_train
print("acc:",np.mean(acc))

