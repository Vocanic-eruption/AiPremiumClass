import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 超参数配置
config = {
    "batch_size": 128,  # 尝试64/256对比效果
    "lr": 0.001,  # 尝试0.01/0.0001对比效果
    "epochs": 15,
    "hidden_dims": [512, 256, 128]  # 调整隐藏层维度
}

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_set = torchvision.datasets.KMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.KMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)


# 可视化样本
def show_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()


show_samples(train_set)


# 构建神经网络模型
class KMNISTModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], output_dim=10):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # 动态生成隐藏层
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # 添加Dropout防止过拟合
            prev_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.network(x)
        return torch.log_softmax(self.fc_out(x), dim=1)


# 初始化模型、损失函数和优化器
model = KMNISTModel(hidden_dims=config["hidden_dims"])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])


# 训练函数
def train(model, device='cpu'):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


# 测试函数
def test(model, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss / len(test_loader), acc


# 训练过程
train_losses = []
test_losses = []
accuracies = []

for epoch in range(1, config["epochs"] + 1):
    train_loss = train(model)
    test_loss, accuracy = test(model)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

    print(f'Epoch: {epoch:02d} | '
          f'Train Loss: {train_loss:.4f} | '
          f'Test Loss: {test_loss:.4f} | '
          f'Accuracy: {accuracy:.2f}%')

# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()
