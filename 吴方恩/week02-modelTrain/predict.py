import numpy as np
from model_train import forward
from sklearn.datasets import load_iris

# 加载数据鸢尾花初始数据
X,y = load_iris(return_X_y=True)

# 取前80-100个数据(未被模型训练过的数据)
idx = np.random.randint(80,100)
print('idx:',idx)
x = X[idx]
y = y[idx]

# 加载训练好的模型参数
params = np.load("model_params.npz")
theta = params['theta']
bias = params['bias']
print(f"加载已训练的模型参数, theta:{theta}, bias:{bias}")

predict = np.round(forward(x, theta, bias))
print(f"验证结果:样本x:{x},真实值y:{y}, 预测值predict:{predict}")
