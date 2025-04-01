import logistic_egression_functions as lrfs
import numpy as np

if __name__ == '__main__':

    # 数据集文件名称
    file_name="datasets_classification.npz"

    # 生成训练数据
    train_X, train_Y, test_X, test_Y = lrfs.make_dataset_classification()

    # 打印训练数据
    print("训练数据集：")
    print("自变量：")
    print(train_X.shape)
    print(train_X)
    print("因变量：")
    print(train_Y.shape)
    print(train_Y)

    print("测试数据集：")
    print("自变量：")
    print(test_X.shape)
    print(test_X)
    print("因变量：")
    print(test_Y.shape)
    print(test_Y)

    # 保存训练数据
    lrfs.save_datasets(train_X, train_Y, test_X, test_Y, file_name)

    # 加载训练数据
    train_A, train_B = lrfs.load_train_data(file_name)
    # 打印加载的训练数据
    print("加载的训练数据集：")
    print("自变量：")
    print(train_A.shape)
    print(train_A)
    print("因变量：")
    print(train_B.shape)
    print(train_B)

    # 加载测试数据
    test_a, test_b = lrfs.load_test_data(file_name)
    # 打印加载的测试数据
    print("加载的测试数据集：")
    print("自变量：")
    print(test_a.shape)
    print(test_a)
    print("因变量：")
    print(test_b.shape)
    print(test_b)

