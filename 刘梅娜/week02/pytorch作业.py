#1. 使用sklearn数据集训练逻辑回归模型
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载数据集
data = load_iris()
X, y = data.data, data.target

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=200)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#2. 调整学习率，样本数据拆分比率，观察训练结果

# 调整学习率
model = LogisticRegression(max_iter=200, C=0.1)  # C是正则化强度的倒数，相当于学习率
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with C=0.1: {accuracy:.2f}")

# 调整样本数据拆分比率
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with test_size=0.3: {accuracy:.2f}")


#3. 训练后模型参数保存到文件，在另一个代码中加载参数实现预测功能

import joblib

# 保存模型
joblib.dump(model, 'logistic_regression_model.pkl')


#加载模型参数并进行预测
# 加载模型
loaded_model = joblib.load('logistic_regression_model.pkl')

# 使用加载的模型进行预测
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Accuracy with loaded model: {accuracy_loaded:.2f}")







