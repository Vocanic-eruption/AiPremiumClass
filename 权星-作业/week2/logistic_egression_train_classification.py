from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

import logistic_egression_functions as lrfs


if __name__ == '__main__':

    # 初始化参数值

    # 权重参数
    theta = lrfs.make_weights()  # 权重
    bias = 0  # 偏差

    # 超参数
    lr = 0.1  # 学习率
    epochs = 3000  # 训练次数

    # 数据集文件名称
    datasets_file_name="datasets_classification.npz"

    # 训练参数文件名称
    weights_bias_file_name="weights_bias_classification.npz"

    # 1.加载训练数据
    train_x, train_y = lrfs.load_train_data(datasets_file_name)

    # 2.训练模型
    theta, bias = lrfs.train(train_x, train_y, theta, bias, lr, epochs)

    # 3.加载测试数据
    test_X, test_y = lrfs.load_test_data(datasets_file_name)

    # 4.模型测试
    idx = np.random.randint(len(test_X))
    x = test_X[idx]
    y = test_y[idx]
    pred = lrfs.predict(x, theta, bias)
    print(f'预测值：{pred}, 真实值：{y}')

    # 5.保存权重参数
    lrfs.save_weights_bias(theta, np.array([bias]), weights_bias_file_name)
 