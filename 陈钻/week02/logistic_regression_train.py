import numpy as np
import pickle

# 加载用于训练的样本集和类别集
def load_train_data(train_sample_set_filepath, train_category_set_filepath):
    train_sample_set = np.load(train_sample_set_filepath)
    train_category_set = np.load(train_category_set_filepath)
    return train_sample_set, train_category_set


# 定义前向传播函数
def forward_propagation(train_sample_set, theta, bias):
    z = np.dot(train_sample_set, theta) + bias
    y_hat = sigmoid(z)
    return y_hat


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 定义损失函数
def loss_function(y, y_hat):
    e = 1e-8
    loss = - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)
    return loss


# 计算梯度
def compute_gradient(train_sample_set, y, y_hat):
    m = train_sample_set.shape[-1]
    d_theta = np.dot(y_hat - y, train_sample_set) / m
    d_bias = np.mean(y_hat - y)
    return d_theta, d_bias


# 训练模型
def train_model(train_sample_set, train_category_set, theta, bias, learning_rate, epochs):
    for epoch in range(epochs):
        y_hat = forward_propagation(train_sample_set, theta, bias)
        loss = np.mean( loss_function(train_category_set, y_hat) )
        if epoch % 100 == 0:
            print(loss_function(train_category_set, y_hat))
            print(f"epoch: {epoch}, loss: {loss}")
        
        d_theta, d_bias = compute_gradient(train_sample_set, train_category_set, y_hat)
        theta = theta - learning_rate * d_theta
        bias = bias - learning_rate * d_bias
    return theta, bias


# 保存模型
def save_model(theta, bias, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump((theta, bias), f)

# 主函数
def main():
    # 初始化参数模型
    theta = np.random.randn(4)
    bias = 0

    # hyperparameters
    learning_rate = 0.01
    epochs = 10000

    # execute training
    train_sample_set, train_category_set = load_train_data("train_sample_set.npy", "train_category_set.npy")
    theta, bias = train_model(train_sample_set, train_category_set, theta, bias, learning_rate, epochs)
    print(f"theta: {theta}, bias: {bias}")
    save_model(theta, bias, "model.pkl")


if __name__ == '__main__':
    main()