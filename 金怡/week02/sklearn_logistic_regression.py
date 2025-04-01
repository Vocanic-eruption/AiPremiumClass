import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')

# 1. 加载数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. 数据拆分：70%训练，20%验证，10%测试
train_size = 0.7
val_size = 0.2
test_size = 0.1
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42)

# 3. 构建流水线：标准化 + 多分类逻辑回归（OneVsRest）
learning_rate = 0.1
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', OneVsRestClassifier(LogisticRegression(C=1/learning_rate, max_iter=1000, solver='lbfgs')))
])
pipeline.fit(X_train, y_train)

# 4. 在验证集和测试集上评估模型
y_val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"验证集准确率: {val_accuracy:.4f}")

y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 5. 计算和绘制ROC曲线
scaler = pipeline.named_steps['scaler']
X_test_scaled = scaler.transform(X_test)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = pipeline.named_steps['clf'].decision_function(X_test_scaled)
n_classes = y_test_bin.shape[1]

plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 6. 保存
joblib.dump(pipeline, "logistic_regression_pipeline.pkl")
print("模型已保存。")

# 7. 加载模型进行预测
def load_and_predict(samples):
    pipeline = joblib.load("logistic_regression_pipeline.pkl")
    predictions = pipeline.predict(samples)
    return predictions

# 使用测试集进行预测
y_test_pred_loaded = load_and_predict(X_test)
print(f"加载后模型的测试集准确率: {accuracy_score(y_test, y_test_pred_loaded):.4f}")


