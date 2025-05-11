import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

# 加载数据
data = pd.read_csv('./weather/Summary of Weather.csv', parse_dates=['Date'], index_col='Date')

# 提取最高气温列
temperature = data['MaxTemp'].values

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_scaled = scaler.fit_transform(temperature.reshape(-1, 1))


# 创建时间序列数据集
def create_dataset(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 10  # 使用过去10天的数据来预测未来气温
X, y = create_dataset(temperature_scaled, seq_length)

# 划分训练集和测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 超参数
input_dim = 1
hidden_dim = 50
output_dim = 1
num_epochs = 100
learning_rate = 0.001

# 转换为 PyTorch 数据集
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型和优化器
model = RNNModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard
writer = SummaryWriter(log_dir='runs/weather_prediction')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar('training loss', running_loss / len(train_loader), epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    writer.add_scalar('test loss', test_loss / len(test_loader), epoch)
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')

writer.close()


def predict(model, inputs, scaler):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = model(inputs)
        predictions = scaler.inverse_transform(outputs.numpy())
    return predictions


# 预测未来1天
last_sequence = X_test[-1].reshape(1, seq_length, 1)
prediction_1_day = predict(model, last_sequence, scaler)
print(f'Predicted temperature for next day: {prediction_1_day[0][0]:.2f}')

# 预测未来5天
predictions_5_days = []
current_sequence = X_test[-1].reshape(1, seq_length, 1)
for _ in range(5):
    prediction = predict(model, current_sequence, scaler)
    predictions_5_days.append(prediction[0][0])
    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, 0] = scaler.transform(prediction.reshape(-1, 1))[0][0]
print(f'Predicted temperatures for next 5 days: {predictions_5_days}')
