import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


torch.manual_seed(42)
np.random.seed(42)
@dataclass
class ModelConfig:
    criterion = nn.CrossEntropyLoss()
    hidden_size = 128
    num_epochs = 100
    learning_rate = 0.001

# 加载 Olivetti Faces 数据集
faces = fetch_olivetti_faces()
X = faces.images  # (400, 64, 64)
y = faces.target  # (400,)

X_tensor = torch.FloatTensor(X) # (400, 1, 64, 64) [batch, channel, height, width]
y_tensor = torch.LongTensor(y)  # (400,)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)


class FaceRNN(nn.Module):
    def __init__(self, input_size=64, hidden_size=ModelConfig.hidden_size, num_classes=40):
        super(FaceRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,  # 每行的像素数（64）
            hidden_size=hidden_size,
            bias=True,
            num_layers=1,
            batch_first=True  # 输入形状为 (batch, seq_len, input_size)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: (batch, seq_len=64, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


class FaceLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=ModelConfig.hidden_size, num_classes=40):
        super(FaceLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 每行64像素
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, 64, 64) → (batch, 64, 64)
        x = x.squeeze(1)
        # LSTM 输出: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class FaceGRU(nn.Module):
    def __init__(self, input_size=64, hidden_size=ModelConfig.hidden_size, num_classes=40):
        super(FaceGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), ModelConfig.learning_rate)
    for epoch in range(ModelConfig.num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = ModelConfig.criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")



def test(model, test_loader, device='cuda'):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)

            _, predicted = torch.max(outputs, 1)  # 直接使用outputs而不是outputs.data

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print(f"测试集准确率: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    writer = SummaryWriter("runs")

    list1 = []

    rnn_model = FaceRNN().to(device)
    train(rnn_model)
    accuracy = test(rnn_model, test_loader, device=device)
    list1.append(accuracy)

    lstm_model = FaceLSTM().to(device)
    train(lstm_model)
    accuracy2 = test(lstm_model, test_loader, device=device)
    list1.append(accuracy2)

    gru_model = FaceGRU().to(device)
    train(gru_model)
    accuracy3 = test(gru_model, test_loader, device=device)
    list1.append(accuracy3)
    writer.close()
    print(list1)