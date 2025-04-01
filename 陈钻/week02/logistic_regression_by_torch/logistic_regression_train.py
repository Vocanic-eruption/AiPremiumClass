import numpy as np
import pickle
import torch

# 加载用于训练的样本集和类别集
def load_train_data(train_sample_set_filepath, train_category_set_filepath):
    train_sample_set = np.load(train_sample_set_filepath)
    train_category_set = np.load(train_category_set_filepath)
    return train_sample_set, train_category_set


# 定义前向传播函数
def forward_propagation(train_sample_set_tensor, theta, bias):
    z = torch.nn.functional.linear(train_sample_set_tensor, theta, bias)
    y_hat = torch.sigmoid(z)
    return y_hat


# 定义损失函数
def loss_function(y, y_hat):
    loss = torch.nn.functional.binary_cross_entropy(y_hat.squeeze(1), y, reduction='mean')
    return loss


# 保存模型
def save_model(theta, bias, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump((theta, bias), f)


# 主函数
def main():
    # 初始化参数模型
    # theta = np.random.randn(4)
    # bias = 0
    theta = torch.randn(1, 4, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)

    # hyperparameters
    learning_rate = 0.01
    epochs = 10000

    for i in range(epochs):
        # execute training
        train_sample_set, train_category_set = load_train_data("train_sample_set.npy", "train_category_set.npy")
        train_sample_set_tensor =  torch.tensor(train_sample_set, dtype=torch.float)
        train_category_set_tensor = torch.tensor(train_category_set, dtype=torch.float)

        r = forward_propagation(train_sample_set_tensor, theta, bias)
        loss = loss_function(train_category_set_tensor, r)

        loss.backward()

        with torch.autograd.no_grad():
            theta -= learning_rate * theta.grad
            bias -= learning_rate * bias.grad
            theta.grad.zero_()
            bias.grad.zero_()

    print(f"theta: {theta}, bias: {bias.item()}")
    save_model(theta, bias, "model.pkl")


if __name__ == '__main__':
    main()