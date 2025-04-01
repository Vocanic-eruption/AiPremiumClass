from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

import logistic_egression_functions as lrfs


if __name__ == '__main__':

    # 初始化参数值
    # 训练参数文件名称
    weights_bias_file_name="weights_bias_classification.npz"

    # 数据集文件名称
    datasets_file_name="datasets_classification.npz"

    # 加载权重参数
    theta, bias = lrfs.load_weights_bias(weights_bias_file_name)

    # 1.加载测试数据
    test_x, test_y = lrfs.load_test_data(datasets_file_name)

    # 2.模型测试
    print("训练集数据：")
    for i in range(10):
        idx = np.random.randint(len(test_x))
        x = test_x[idx]
        y = test_y[idx]
        pred = lrfs.predict(x, theta, bias)
        print(f'预测值：{pred}, 真实值：{y}, 正确性：{(pred==y)}')


    # 3.使用新数据测试

    # 生成训练数据
    train_X, train_Y, test_X, test_Y = lrfs.make_dataset_classification()
    
    print("\n新生成训练集数据：")
    for i in range(10):
        idx = np.random.randint(len(train_X))
        x = train_X[idx]
        y = train_Y[idx]
        pred = lrfs.predict(x, theta, bias)
        print(f'预测值：{pred}, 真实值：{y}, 正确性：{(pred==y)}')

    print("新生成测试集数据：")
    for i in range(10):
        idx = np.random.randint(len(test_X))
        x = test_X[idx]
        y = test_Y[idx]
        pred = lrfs.predict(x, theta, bias)
        print(f'预测值：{pred}, 真实值：{y}, 正确性：{(pred==y)}')
        