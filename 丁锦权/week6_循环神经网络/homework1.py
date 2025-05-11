import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# 获取数据和标签
olivetti_faces = fetch_olivetti_faces(data_home='./faceData', shuffle=True)
X = olivetti_faces.data
y = olivetti_faces.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 将数据转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义基础RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 定义BiRNN模型
class BiRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers * 2, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 定义训练方法
def train(model, train_loader, test_loader, num_epochs, writer, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 64, 64).to(device)  # 将图像展平为序列
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar(f'{model_name}/training loss', loss.item(), epoch * len(train_loader) + i)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # 测试准确率
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, 64, 64).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            writer.add_scalar(f'{model_name}/accuracy', accuracy, epoch)
            print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    writer.close()


if __name__ == '__main__':
    # 定义模型参数
    input_size = 64  # 每个时间步的输入大小
    hidden_size = 128  # 隐藏层大小
    num_layers = 2  # RNN 层数
    num_classes = 40  # Olivetti Faces 数据集有 40 个类别

    # 创建 TensorBoard 记录器
    writer_rnn = SummaryWriter(log_dir='runs/RNN')
    writer_lstm = SummaryWriter(log_dir='runs/LSTM')
    writer_gru = SummaryWriter(log_dir='runs/GRU')
    writer_birnn = SummaryWriter(log_dir='runs/BiRNN')

    # 训练不同模型
    train(RNNClassifier(input_size, hidden_size, num_layers, num_classes), train_loader, test_loader, 10, writer_rnn,
          'RNN')
    train(LSTMClassifier(input_size, hidden_size, num_layers, num_classes), train_loader, test_loader, 10, writer_lstm,
          'LSTM')
    train(GRUClassifier(input_size, hidden_size, num_layers, num_classes), train_loader, test_loader, 10, writer_gru,
          'GRU')
    train(BiRNNClassifier(input_size, hidden_size, num_layers, num_classes), train_loader, test_loader, 10,
          writer_birnn, 'BiRNN')
