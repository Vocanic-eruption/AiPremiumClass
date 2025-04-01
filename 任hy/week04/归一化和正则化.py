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
WEIGHT_DECAY = 1e-5  # L2 正则化参数

# 定义神经网络模型
class OlivettiModel(nn.Module):
    def __init__(self):
        super(OlivettiModel, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 128)
        self.bn1 = nn.BatchNorm1d(128)  # 归一化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)  # Dropout 正则化
        self.fc2 = nn.Linear(128, 40)  # 40 个类别

    def forward(self, x):
        x = x.view(-1, 64 * 64)  # 将图像展平
        x = self.fc1(x)
        x = self.bn1(x)  # 归一化
        x = self.relu(x)
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)
        return x

# 定义训练和测试类
class Trainer:
    def __init__(self, model, train_data, test_data, batch_size, lr, epochs):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_data, batch_size=batch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)  # L2 正则化
        self.train_losses = []  # 记录训练损失
        self.test_losses = []  # 记录测试损失

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            for data, target in self.train_dl:
                # 前向运算
                output = self.model(data)
                # 计算损失
                loss = self.loss_fn(output, target)
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / len(self.train_dl)
            self.train_losses.append(avg_train_loss)

            # 验证损失
            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for data, target in self.test_dl:
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    epoch_test_loss += loss.item()

            # 计算平均验证损失
            avg_test_loss = epoch_test_loss / len(self.test_dl)
            self.test_losses.append(avg_test_loss)

            print(f'Epoch:{epoch + 1}, Train Loss:{avg_train_loss:.4f}, Test Loss:{avg_test_loss:.4f}')

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

# 数据集加载
train_data = OlivettiFaces(root='./olivetti_data', transform=ToTensor(), download=True)
test_data = OlivettiFaces(root='./olivetti_data', transform=ToTensor(), download=True)

# 初始化模型和训练器
model = OlivettiModel()
trainer = Trainer(model, train_data, test_data, BATCH_SIZE, LR, EPOCHS)

# 训练和测试
trainer.train()
trainer.plot_loss()
