import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import numpy as np


# hyperparameters
learning_rate = 1e-2
batch_size = 8
epochs = 20


# 加载用于训练的样本集和类别集
def load_train_data(train_sample_set_filepath, train_category_set_filepath):
    train_sample_set = np.load(train_sample_set_filepath)
    train_category_set = np.load(train_category_set_filepath)
    return train_sample_set, train_category_set


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # self.linear_sigmoid_stack = nn.Sequential(
        #     nn.Linear(64*64, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(0.1),
        #     nn.Sigmoid(),
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(0.1),
        #     nn.Sigmoid(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.1),
        #     nn.Sigmoid(),
        #     nn.Linear(512, 40),
        # )
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Sigmoid(),
            nn.Linear(512, 40),
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits


# 主函数
def main():
    train_sample_set, train_category_set = load_train_data("train_sample_set.npy", "train_category_set.npy")
    train_sample_tensor = torch.tensor(train_sample_set, dtype=torch.float32)
    train_category_tensor = torch.tensor(train_category_set, dtype=torch.long)
    training_data = [(train_sample_tensor[i], train_category_tensor[i]) for i in range(len(train_sample_tensor))]

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用 {device} 设备")

    model = NeuralNetwork().to(device)
    print(model)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


    model.train()
    for epoch in range(epochs):
        for data, target in train_dataloader:
            # print(data.shape)
            # print(target.shape)
            pred = model(data.reshape(data.shape[0], -1).to(device))
            loss = loss_fn(pred, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch} Loss: {loss.item()}")


    # 保存模型参数
    torch.save(model.state_dict(), 'model_weights.pth')

    # print(f"theta: {theta}, bias: {bias}")
    # save_model(theta, bias, "model.pkl")


if __name__ == '__main__':
    main()

