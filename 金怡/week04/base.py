import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
matplotlib.use('TkAgg')
# 1. 加载Olivetti Faces数据集
data = fetch_olivetti_faces(shuffle=True)
X = data.images  # (400, 64, 64)
y = data.target  # (400,)

# 2. 数据归一化 & 预处理
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
Y = torch.tensor(y, dtype=torch.long)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. 创建DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 4. 搭建神经网络模型（带正则化）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 40)
        self.dropout = nn.Dropout(0.5)  # Dropout正则化
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 5. 训练函数
def train_model(optimizer_choice):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_choice(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则化
    epochs = 20
    loss_values = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return loss_values


# 6. 运行不同优化器实验
optimizers = {'SGD': optim.SGD, 'Adam': optim.Adam, 'RMSprop': optim.RMSprop}
loss_results = {}

for opt_name, opt in optimizers.items():
    print(f"Training with {opt_name}...")
    loss_results[opt_name] = train_model(opt)

# 7. 可视化Loss变化
plt.figure(figsize=(10, 5))
for opt_name, losses in loss_results.items():
    plt.plot(losses, label=opt_name)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison for Different Optimizers")
plt.legend()
plt.show()
