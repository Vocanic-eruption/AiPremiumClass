import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# hyperparameters
batch_size = 64

testing_data = datasets.FashionMNIST(
    root="data", 
    train=False, 
    download=True, 
    transform=ToTensor(), 
)

test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用 {device} 设备")


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10),
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits


# 加载时需先创建模型实例，再加载参数
loaded_model = NeuralNetwork()  # 需要先定义相同的模型类
loaded_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device(device)))
# loaded_model.load_state_dict(torch.load('model_weights.pth'))
loaded_model.eval()  # 设置为评估模式（如涉及 Dropout/BatchNorm）

print(loaded_model)

# 测试
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dataloader:
        # output = loaded_model(data.reshape(data.shape[0], -1).to(device))
        output = loaded_model(data.reshape(data.shape[0], -1))
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f"Accuracy: {100 * correct / total}%")