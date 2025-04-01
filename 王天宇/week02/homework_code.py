
from sklearn.datasets import load_iris # 引入数据集
from logistic_regression import LogisticRegression # 引入模型函数
from sklearn.model_selection import train_test_split # 引入数据集划分函数 
import numpy as np
X,y = load_iris(return_X_y=True)

# 准备样本数据
x_arr = X[:100] #(100, 4)
y_arr = y[:100] #(100,)

# x_arr = X
# y_arr = y

print(x_arr.shape)
print(y_arr.shape)

X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.3)
# print(X_train)
# print(y_train)

model = LogisticRegression(0.0005,10000) #学习率，次数
model.fit(X_train, y_train)

idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]
y_pred = model.predict(x)
print(f"y: {y}, predict: {y_pred}")

#结果记录（只取0,1两个特征）：
# 0 的 sigmoid 为 0.4左右
# 1 的 sigmoid 为 0.6左右

#结果记录（取0,1,2三个特征）：
# y: 0, predict: 0.99
# y: 1, predict: 0.99
# y: 2, predict: 0.99

# 问题总结：
#   特征和样本数据不匹配的时候，sigmoid不会到最大值
#   在0.4、0.6左右（只针对三个特征，n个特征的情况下待确定）徘徊

#引出：
# 如何确定特征？
# 1. 画出散点图，观察特征与标签的关系
# 2. 尝试删除特征，看是否影响结果