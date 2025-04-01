import torch
from sklearn.datasets import make_classification


# 创建张量
X, y = make_classification(n_features=20)
tensor_x = torch.tensor(X, dtype=torch.float)


# 加载保存的模型参数
checkpoint = torch.load('model_params.pth')
w = checkpoint['w']
b = checkpoint['b']


# 使用加载的参数进行预测
with torch.no_grad():
    # 向前运算
    r = torch.nn.functional.linear(tensor_x, w, b)
    r = torch.sigmoid(r)


# 输出预测结果
print("Predictions:", r)



"""

知识点整理：

1.逻辑回归是一种线性分类器，通过将线性模型的输出通过 Sigmoid 函数映射到 0 到 1 之间的概率来进行二分类。
2.训练过程中，使用交叉熵损失函数来衡量预测值与真实标签之间的差异，并通过梯度下降法更新模型参数。
3.逻辑回归通过最大化似然函数或最小化损失函数（比如交叉熵损失函数）来学习模型参数


"""
