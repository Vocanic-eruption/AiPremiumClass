import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# 1. 加载鸢尾花数据集（使用所有 4 个特征）
iris = datasets.load_iris()
X, y = iris.data, iris.target  # 全部特征
class_names = iris.target_names

# 2. 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 设定超参数
test_size = 0.2  # 测试集比例
learning_rate = 0.1  # 逻辑回归 C = 1 / learning_rate
max_iter = 200  # 训练迭代次数

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 5. 训练逻辑回归模型
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1/learning_rate, max_iter=max_iter)

# 记录 Loss 和 Accuracy
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for i in range(1, max_iter + 1):
    model.max_iter = i
    model.fit(X_train, y_train)
    
    # 计算 Loss
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    test_loss = log_loss(y_test, model.predict_proba(X_test))
    
    # 计算 Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # 记录数据
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# 6. 评估模型
final_test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Final Test Accuracy: {final_test_acc:.4f}")

# 7. 保存模型和标准化参数
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 8. 绘制 Loss 变化曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, max_iter + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, max_iter + 1), test_losses, label="Test Loss", color="red")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Training vs. Testing Loss")
plt.legend()

# 9. 绘制 Accuracy 变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, max_iter + 1), train_accs, label="Train Accuracy", color="blue")
plt.plot(range(1, max_iter + 1), test_accs, label="Test Accuracy", color="red")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Training vs. Testing Accuracy")
plt.legend()

plt.show()
