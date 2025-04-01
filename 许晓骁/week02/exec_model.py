import numpy as np
import joblib  # 用于加载模型

# 1. 加载模型参数
model = joblib.load('logistic_regression_model.pkl')

# 2. 准备新的数据进行预测
# 这里使用示例数据，实际应用中应根据需要提供新的输入数据
new_data = np.array([[5.1, 3.5, 1.4, 0.2],  # 示例数据
                     [6.7, 3.1, 4.7, 1.5]])

# 3. 进行预测
predictions = model.predict(new_data)

# 4. 输出预测结果
print(f'预测结果: {predictions}')
