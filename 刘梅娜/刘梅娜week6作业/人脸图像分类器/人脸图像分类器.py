import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 加载数据集
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义RNN模型
class RNNModel(nn.Module):
    # 定义RNN模型类，继承自nn.Module
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # 初始化函数，传入输入大小、隐藏层大小、层数和类别数
        super(RNNModel, self).__init__()
        # 调用父类的初始化函数
        self.hidden_size = hidden_size
        # 定义隐藏层大小
        self.num_layers = num_layers
        # 定义层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 定义RNN层，输入大小为input_size，隐藏层大小为hidden_size，层数为num_layers，batch_first为True
        self.fc = nn.Linear(hidden_size, num_classes)

        # 定义全连接层，输入大小为hidden_size，输出大小为num_classes
    def forward(self, x):
        # 定义前向传播函数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 初始化隐藏状态，大小为(num_layers, batch_size, hidden_size)，并将其移动到与输入相同的设备上
        out, _ = self.rnn(x, h0)
        # 将输入和隐藏状态传入RNN层，得到输出和新的隐藏状态
        out = self.fc(out[:, -1, :])
        # 将输出的最后一个时间步的隐藏状态传入全连接层，得到最终的输出
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # 初始化LSTM模型
        super(LSTMModel, self).__init__()
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置LSTM层数
        self.num_layers = num_layers
        # 定义LSTM层，输入大小为input_size，隐藏层大小为hidden_size，层数为num_layers，batch_first=True表示输入数据的第一个维度为batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，输入大小为hidden_size，输出大小为num_classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 将输入x和初始状态传入LSTM，得到输出out和新的状态
        out, _ = self.lstm(x, (h0, c0))
        # 取出最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        # 返回输出
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # 初始化GRU模型
        super(GRUModel, self).__init__()
        # 定义隐藏层大小
        self.hidden_size = hidden_size
        # 定义GRU层数
        self.num_layers = num_layers
        # 定义GRU层，输入大小为input_size，隐藏层大小为hidden_size，GRU层数为num_layers，batch_first=True表示输入数据的第一个维度为batch_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，输入大小为hidden_size，输出大小为num_classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态h0，大小为(num_layers, x.size(0), self.hidden_size)，并将其移动到x所在的设备上
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 将输入x和初始化的隐藏状态h0传入GRU模型，得到输出out和新的隐藏状态_
        out, _ = self.gru(x, h0)
        # 将GRU的输出out的最后一个时间步的输出传入全连接层fc，得到最终的输出out
        out = self.fc(out[:, -1, :])
        # 返回最终的输出out
        return out

# 训练函数
def train_model(model, train_loader, criterion, optimizer, writer, num_epochs=25):
    # 将模型设置为训练模式
    model.train()
    # 遍历每个epoch
    for epoch in range(num_epochs):
        # 初始化每个epoch的损失和正确预测数量
        running_loss = 0.0
        correct = 0
        total = 0
        # 遍历每个batch
        for images, labels in train_loader:
            images = images.view(images.size(0), -1, 64)  # Reshape for RNN
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# 测试函数
def test_model(model, test_loader, criterion, writer):
    # 将模型设置为评估模式
    model.eval()
    # 初始化损失和正确预测的数量
    running_loss = 0.0
    correct = 0
    total = 0
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据集
        for images, labels in test_loader:
            images = images.view(images.size(0), -1, 64)  # Reshape for RNN
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    writer.add_scalar('Loss/test', epoch_loss, 0)
    writer.add_scalar('Accuracy/test', epoch_acc, 0)
    print(f'Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}')

# 主函数
def main():
    # 定义输入层大小
    input_size = 64
    # 定义隐藏层大小
    hidden_size = 128
    # 定义层数
    num_layers = 2
    # 定义类别数
    num_classes = 40
    # 定义训练轮数
    num_epochs = 25
    # 定义学习率
    learning_rate = 0.001

    # 定义模型
    models = {
        'RNN': RNNModel(input_size, hidden_size, num_layers, num_classes),
        'LSTM': LSTMModel(input_size, hidden_size, num_layers, num_classes),
        'GRU': GRUModel(input_size, hidden_size, num_layers, num_classes)
    }

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 遍历模型
    for name, model in models.items():
        # 打印训练模型
        print(f'Training {name} model...')
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # 定义写入器
        writer = SummaryWriter(f'runs/{name}')
        # 训练模型
        train_model(model, train_loader, criterion, optimizer, writer, num_epochs)
        # 测试模型
        test_model(model, test_loader, criterion, writer)
        # 关闭写入器
        writer.close()

if __name__ == '__main__':
    main()