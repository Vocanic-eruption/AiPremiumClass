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
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out

def load_weather_data(seq_length=14, pred_days=5):
    data = pd.read_csv('d:\datasets\Summary of Weather.csv', parse_dates=['Date'])
    temp = data['MaxTemp'].values.reshape(-1, 1)

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


if __name__ == '__main__':
    run_file_path = r"d:\vsCodeProj\AiPremiumClass\王健钢\week6_循环神经网络\run\weather"
    writer = SummaryWriter(log_dir=run_file_path)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    dataset, scaler = load_weather_data()
    train_size = int(0.8 * len(dataset))

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = WeatherRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        # 预测
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer.add_scalar('test accuracy', accuracy, epoch)


    writer.close()