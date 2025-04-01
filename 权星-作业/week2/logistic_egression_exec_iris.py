from sklearn.datasets import load_iris
import numpy as np

import logistic_egression_functions as lrfs


if __name__ == "__main__":

    # 初始化参数值

    # 训练参数文件名称
    weights_bias_file_name="weights_bias_iris.npz"

    # 加载权重参数
    theta, bias = lrfs.load_weights_bias(weights_bias_file_name)

    # 打印权重参数
    print('打印参数：')
    print(f'theta:{theta}')
    print(f'bias:{bias}')

    # 1.加载鸢尾花数据集
    train_X, train_Y, test_X, test_Y = lrfs.load_dataset_iris(0.5)

    # 2.模型测试
    print('改变测试数据比例:')
    print("训练集数据：")
    for i in range(10):
        idx = np.random.randint(len(train_X))
        x = train_X[idx]
        y = train_Y[idx]
        pred = lrfs.predict(x, theta, bias)
        print(f'预测值：{pred}, 真实值：{y}, 正确性：{(pred==y)}')

    print("测试集数据：")
    for i in range(10):
        idx = np.random.randint(len(test_X))
        x = test_X[idx]
        y = test_Y[idx]
        pred = lrfs.predict(x, theta, bias)
        print(f'预测值：{pred}, 真实值：{y}, 正确性：{(pred==y)}')

    
