import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# 载入数据集
X, y = load_iris(return_X_y=True)
# 只取前100个样本(类别0和类别1)
X = X[:100]
y = y[:100]  # 这样y只包含0和1两个类别

# 对数据进行描述性统计
print(f"数据集形状: {X.shape}")
print(f"标签分布: {np.bincount(y)}")

# ----- 逻辑回归模型原理 -----
# 逻辑回归是一种用于二分类的线性模型。虽然名称包含"回归"，但它实际上是一种分类算法。
# 基本原理：
# 1. 线性部分: z = w0 + w1*x1 + w2*x2 + ... + wn*xn (类似线性回归)
# 2. 非线性转换: 使用sigmoid函数 σ(z) = 1/(1+e^(-z)) 将线性输出转换为0-1之间的概率值
# 3. 决策规则: 如果 σ(z) >= 0.5，预测为类别1；否则预测为类别0
#
# 模型训练通过最大化似然函数或最小化对数损失(交叉熵)来优化参数w
# 损失函数: -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
#
# 使用随机梯度下降法等优化算法迭代更新参数w


# 设定不同的测试集比例和学习率进行实验
test_sizes = [0.2, 0.3, 0.4]
learning_rates = [0.01, 0.1, 1.0, 10.0]

results = {}

for test_size in test_sizes:
    for learning_rate in learning_rates:
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 创建并训练逻辑回归模型
        # C参数是正则化强度的倒数，值越小正则化越强
        # solver='liblinear'适用于小数据集
        model = LogisticRegression(
            C=1 / learning_rate,  # C值越小，正则化强度越大，对应较大的学习率
            solver='liblinear',  # 使用liblinear求解器
            random_state=42,
            max_iter=1000  # 增加最大迭代次数以确保收敛
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上评估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 存储结果
        key = f"test_size={test_size}, learning_rate={learning_rate}"
        results[key] = {
            'accuracy': accuracy,
            'model': model,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        print(f"\n--- 参数: {key} ---")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print(f"模型参数(权重): {model.coef_}")
        print(f"模型截距: {model.intercept_}")
        print(f"准确率: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

# 找出最佳模型
best_model_key = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_key]['model']
print(f"\n最佳模型: {best_model_key}, 准确率: {results[best_model_key]['accuracy']:.4f}")

# 保存最佳模型参数到文件 (使用NumPy的savez方法)
# 提取模型的权重(theta)和偏置(bias)
theta = best_model.coef_  # 权重系数
bias = best_model.intercept_  # 偏置项/截距

# 保存参数到npz文件
model_params_filename = 'model_params.npz'
np.savez(model_params_filename, theta=theta, bias=bias)
print(f"最佳模型参数已保存至 {model_params_filename}")
