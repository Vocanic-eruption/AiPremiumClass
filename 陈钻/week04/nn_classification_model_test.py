import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


# hyperparameters
batch_size = 8

# 加载用于测试的样本集和类别集
def load_test_data(test_sample_set_filepath, test_category_set_filepath):
    test_sample_set = np.load(test_sample_set_filepath)
    test_category_set = np.load(test_category_set_filepath)
    return test_sample_set, test_category_set


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
    test_sample_set, test_category_set = load_test_data("test_sample_set.npy", "test_category_set.npy")
    test_sample_tensor = torch.tensor(test_sample_set, dtype=torch.float32)
    test_category_tensor = torch.tensor(test_category_set, dtype=torch.long)
    test_data = [(test_sample_tensor[i], test_category_tensor[i]) for i in range(len(test_sample_tensor))]
    
    print("-----------------test_category_set---------------------")
    print(test_category_set)
    print(test_category_set.shape)
    print("-------------------test_sample_set-------------------")
    print(test_sample_set)
    print(test_sample_set.shape)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用 {device} 设备")

    # 加载时需先创建模型实例，再加载参数
    loaded_model = NeuralNetwork()  # 需要先定义相同的模型类
    loaded_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device)))
    loaded_model.eval()  # 设置为评估模式（如涉及 Dropout/BatchNorm）
    print(loaded_model)

    # 测试
    print("-------------------Accuracy Output-------------------")
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = loaded_model(data.reshape(data.shape[0], -1))
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {100 * correct / total}%")


if __name__ == '__main__':
    main()