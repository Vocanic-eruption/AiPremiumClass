import numpy as np
import pickle
import torch

# 加载用于测试的样本集和类别集
def load_test_data(test_sample_set_filepath, test_category_set_filepath):
    test_sample_set = np.load(test_sample_set_filepath)
    test_category_set = np.load(test_category_set_filepath)
    return test_sample_set, test_category_set


# 加载模型
def load_model(model_filepath):
    with open(model_filepath, 'rb') as f:
        theta, bias = pickle.load(f)
    return theta, bias


# 定义前向传播函数
def forward_propagation(train_sample_set_tensor, theta, bias):
    z = torch.nn.functional.linear(train_sample_set_tensor, theta, bias)
    y_hat = torch.sigmoid(z)
    return y_hat


# 定义推理函数
def reasoning(test_sample_tensor, theta, bias):
    y_hat = forward_propagation(test_sample_tensor, theta, bias)
    y_pred = np.where(y_hat >= 0.5, 1, 0)
    return y_pred


# 主函数
def main():
    theta, bias = load_model('model.pkl')
    print(f"theta: {theta}, bias: {bias}")

    test_sample_set, test_category_set = load_test_data('test_sample_set.npy', 'test_category_set.npy')
    test_sample_set_tensor = torch.tensor(test_sample_set, dtype=torch.float)
    test_category_set_tensor = torch.tensor(test_category_set, dtype=torch.float)

    print(f"test_sample_set_tensor: {test_sample_set_tensor}, test_category_set_tensor: {test_category_set_tensor}")

    y_pred = reasoning(test_sample_set_tensor[2], theta, bias)
    print(f"y_pred: {y_pred}")


if __name__ == '__main__':
    main()