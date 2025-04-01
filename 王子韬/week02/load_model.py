import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 加载保存的模型参数
model_params_filename = 'best_iris_model.npz'
loaded_params = np.load(model_params_filename)
theta = loaded_params['theta']
bias = loaded_params['bias']

print(f"成功加载模型参数")
print(f"模型参数(权重): {theta}")
print(f"模型截距: {bias}")

# 使用加载的参数创建新的逻辑回归模型
loaded_model = LogisticRegression()
# 手动设置模型参数
loaded_model.coef_ = theta
loaded_model.intercept_ = bias
loaded_model.classes_ = np.array([0, 1])  # 设置类别标签

# 加载一些数据用于测试预测
X, y = load_iris(return_X_y=True)
X_test = X[:100]  # 使用前100个样本作为测试
y_test = y[:100]  # 对应的实际标签

# 使用加载的模型进行预测
y_pred = loaded_model.predict(X_test)
probabilities = loaded_model.predict_proba(X_test)  # 预测为每个类别的概率

# 输出一些样本的预测结果
for i in range(10):  # 只展示前10个样本
    # 获取概率值
    prob_class_0 = probabilities[i, 0]
    prob_class_1 = probabilities[i, 1]

    print(f"\n样本 {i + 1}:")
    print(f"  特征值: {X_test[i]}")
    print(f"  真实类别: {y_test[i]}")
    print(f"  预测类别: {y_pred[i]}")
    print(f"  预测为类别0的概率: {prob_class_0:.4f}")
    print(f"  预测为类别1的概率: {prob_class_1:.4f}")

# 评估整体准确率
accuracy = (y_pred == y_test).mean()
print(f"\n整体准确率: {accuracy:.4f}")