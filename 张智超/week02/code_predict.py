from code_datasets import datasets_path
from code_train import param_path, forward
import numpy as np
import os

def predict():
    x = np.load(os.path.join(datasets_path, 'X_test.npy'))
    y = np.load(os.path.join(datasets_path, 'y_test.npy'))
    theta = np.load(os.path.join(param_path, 'theta.npy'))
    bias = np.load(os.path.join(param_path, 'bias.npy'))
    num = 0
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        pred = np.round(forward(xi, theta, bias))
        res = (pred == yi).item()
        print(f'i={i},真实值：{yi},预测值：{pred}, 预测结果：{res}')
        if res: num += 1
    print('======================')
    print(f'共{len(x)}次预测，预测准确率：{(num / len(x)) * 100}%')

# 数据预测
predict()