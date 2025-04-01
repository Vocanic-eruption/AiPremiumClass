from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # 用于保存和加载模型

# 1. 加载数据集
# 使用 scikit-learn 提供的鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
y = iris.target  # 目标数据（花的种类）

# 2. 拆分数据集
# 将数据集拆分为训练集和测试集
# test_size = 0.2 表示 20% 的数据用于测试，80% 用于训练
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 3. 训练逻辑回归模型
# 创建逻辑回归模型实例
# C参数控制正则化强度，值越小表示越强的正则化
learning_rate = 1.0  # 这里设置学习率（C的倒数）
model = LogisticRegression(C=learning_rate, max_iter=200)  # max_iter 设置最大迭代次数
model.fit(X_train, y_train)  # 训练模型

# 4. 预测并评估模型
y_pred = model.predict(X_test)  # 使用测试集进行预测
accuracy = accuracy_score(y_test, y_pred)  # 计算模型准确率
print(f'模型准确率: {accuracy:.2f}')  # 输出准确率

# 5. 保存模型参数到文件
joblib.dump(model, 'logistic_regression_model.pkl')  # 将训练好的模型保存到文件
