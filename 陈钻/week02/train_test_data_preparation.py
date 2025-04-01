from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


# 加载样本数据， 形成样本集和类别集的全量集合
def load_sampleing_data():
    X, y = load_iris(return_X_y=True)
    sample_set_A = X[0:50]
    sample_set_B = X[50:100]
    sample_set = np.concatenate((sample_set_A, sample_set_B), axis=0)

    category_A = y[0:50]
    category_B = y[50:100]
    category_set = np.concatenate((category_A, category_B), axis=0)
    return sample_set, category_set


# 从全量集合中， 划分训练集和测试集，以test_size为测试集的比例
def get_train_test_data(sample_set, category_set, test_size=0.2):
    train_sample_set, test_sample_set, train_category_set, test_category_set = train_test_split(sample_set, category_set, test_size=test_size, random_state=42, shuffle=True)
    return train_sample_set, test_sample_set, train_category_set, test_category_set


# 保存用于训练的样本集和类别集
def save_train_data(train_sample_set, train_category_set):
    np.save('train_sample_set.npy', train_sample_set)
    np.save('train_category_set.npy', train_category_set)


# 保存用于测试的样本集和类别集
def save_test_data(test_sample_set, test_category_set):
    np.save('test_sample_set.npy', test_sample_set)
    np.save('test_category_set.npy', test_category_set)


# 主函数
def main():
    sample_set, category_set = load_sampleing_data()
    train_sample_set, test_sample_set, train_category_set, test_category_set = get_train_test_data(sample_set, category_set, 0.3)
    save_train_data(train_sample_set, train_category_set)
    save_test_data(test_sample_set, test_category_set)


if __name__ == '__main__':
    main()