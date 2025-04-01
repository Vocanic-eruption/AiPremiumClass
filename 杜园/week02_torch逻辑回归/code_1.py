
import torch
import matplotlib.pyplot as plt

# 样本数据
x_data = torch.linspace(-1, 6, 120)
# 模拟损失函数
y_data = (x_data - 2.5) ** 2 - 1
# 绘制曲线图
plt.plot(x_data, y_data)
plt.show()