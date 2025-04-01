from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import os

# 创建文件夹保存数据集
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
datasets_path = os.path.join(current_directory, 'datasets')
if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)
def save_datasets(test_size):
    X,y = load_iris(return_X_y=True)
    X = X[:100]
    y = y[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # 将数据保存到datasets文件夹下
    if not os.path.exists(datasets_path):
        os.mkdir(datasets_path)
    np.save(os.path.join(datasets_path, 'X_train'), X_train)
    np.save(os.path.join(datasets_path, 'X_test'), X_test)
    np.save(os.path.join(datasets_path, 'y_train'), y_train)
    np.save(os.path.join(datasets_path, 'y_test'), y_test)

if __name__ == "__main__":
    print('==生成训练数据和测试数据(test_size: 数据拆分比率)==')
    save_datasets(test_size=0.2)