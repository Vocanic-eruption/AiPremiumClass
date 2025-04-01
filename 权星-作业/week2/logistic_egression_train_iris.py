from sklearn.datasets import load_iris
import numpy as np

import logistic_egression_functions as lrfs


if __name__ == "__main__":

    # 初始化参数值

    # 权重参数
    theta = lrfs.make_weights(4)  # 权重
    bias = 0  # 偏差

    # 超参数
    lr = 0.1  # 学习率
    epochs = 3000  # 训练次数
    
    # 训练参数文件名称
    weights_bias_file_name="weights_bias_iris.npz"

    # 1.加载鸢尾花数据集
    train_X, train_Y, test_X, test_Y = lrfs.load_dataset_iris()

    # 2.训练模型
    theta, bias = lrfs.train(train_X, train_Y, theta, bias, lr, epochs)

    # 4.模型测试
    idx = np.random.randint(len(test_X))
    x = test_X[idx]
    y = test_Y[idx]
    pred = lrfs.predict(x, theta, bias)
    print(f'预测值：{pred}, 真实值：{y}')

    # 5.保存权重参数
    lrfs.save_weights_bias(theta, np.array([bias]), weights_bias_file_name)

    
