import joblib
import numpy as np

# 1. 加载训练好的模型和标准化器
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. 创建新样本（使用全部 4 个特征）
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # 需要预测的样本

# 3. 进行标准化
new_sample_scaled = scaler.transform(new_sample)

# 4. 进行预测
prediction = model.predict(new_sample_scaled)
probabilities = model.predict_proba(new_sample_scaled)

class_names = ['setosa', 'versicolor', 'virginica']
print(f"Predicted Class: {class_names[prediction[0]]}")
print(f"Class Probabilities: {probabilities}")
