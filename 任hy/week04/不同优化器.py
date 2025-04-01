import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import OlivettiFaces
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义超参数
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-5

# 定义神经网络模型
class OlivettiModel(nn.Module):
    def __init__(self):
        super(OlivettiModel, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义训练类（支持自定义优化器）
class Trainer:
    def __init__(self, model, train_data, test_data, batch_size, optimizer, epochs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.optimizer = optimizer  # 直接传入优化器
        self.epochs = epochs
        self.train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_data, batch_size=batch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.test_losses = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for data, target in self.train_dl:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(self.train_dl)
            self.train_losses.append(avg_train_loss)

            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for data, target in self.test_dl:
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    epoch_test_loss += loss.item()

            avg_test_loss = epoch_test_loss / len(self.test_dl)
            self.test_losses.append(avg_test_loss)

            print(f'Epoch:{epoch + 1}, Train Loss:{avg_train_loss:.4f}, Test Loss:{avg_test_loss:.4f}')

# 数据集加载
train_data = OlivettiFaces(root='./olivetti_data', transform=ToTensor(), download=True)
test_data = OlivettiFaces(root='./olivetti_data', transform=ToTensor(), download=True)

# 定义优化器配置
optimizers = {
    "Adam": optim.Adam,
    "SGD": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),  # 调整学习率和动量
    "RMSprop": lambda params: optim.RMSprop(params, lr=1e-3, alpha=0.99)
}

# 存储不同优化器的训练结果
results = {}

# 测试不同优化器
for opt_name, opt_func in optimizers.items():
    print(f"\n=== Training with {opt_name} ===")
    # 初始化模型和优化器
    model = OlivettiModel()
    optimizer = opt_func(model.parameters(), weight_decay=WEIGHT_DECAY)  # 统一添加 L2 正则化
    trainer = Trainer(model, train_data, test_data, BATCH_SIZE, optimizer, EPOCHS)
    trainer.train()
    results[opt_name] = {
        "train_loss": trainer.train_losses,
        "test_loss": trainer.test_losses
    }

# 绘制对比图
plt.figure(figsize=(12, 6))

# 绘制训练损失
plt.subplot(1, 2, 1)
for opt_name, data in results.items():
    plt.plot(data["train_loss"], label=f"{opt_name} Train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()

# 绘制测试损失
plt.subplot(1, 2, 2)
for opt_name, data in results.items():
    plt.plot(data["test_loss"], label=f"{opt_name} Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss Comparison")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
