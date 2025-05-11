import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=5):
        super(WeatherRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=5,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # out shape: [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])  # 取最后一个时间步 -> [batch, output_size]
        return out
def load_weather_data(seq_length=14, pred_days=5):
    data = pd.read_csv('E:\\workspacepython\\AiPremiumClass\\week6\\Summary of Weather.csv', parse_dates=['Date'])
    temp = data['MaxTemp'].values.reshape(-1, 1) # 提取最高温度列并转换为numpy数组

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(temp)

    X, y = [], []
    for i in range(len(scaled)-seq_length-pred_days+1):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length:i+seq_length+pred_days].reshape(-1))

    X = np.array(X).astype(np.float32)  # shape: [n, 14, 1]
    y = np.array(y).astype(np.float32)  # shape: [n, 5]

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return dataset, scaler

def train_weather():
    writer = SummaryWriter(log_dir='runs/weather_lstm')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, scaler = load_weather_data()
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = WeatherRNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.to(device))
                test_loss += criterion(outputs, targets.to(device)).item()

        test_loss /= len(test_loader)
        writer.add_scalars('Loss', {'train': loss.item(), 'test': test_loss}, epoch)
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')

    writer.close()

if __name__ == '__main__':
    train_weather()