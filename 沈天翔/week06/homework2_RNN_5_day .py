import os
from tkinter import W
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
import shutil
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

# 定义模型参数
class RNN_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, outpoust_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,   # x的特征维度
            hidden_size=hidden_size,  # 隐藏层神经元数量
            bias=True,        # 偏置
            num_layers=num_layers,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(hidden_size, outpoust_size)  # 输出层

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out

# 将数据重塑为适合RNN输入的形状
def create_sequences(data, sequence_length, prediction_length=5):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - prediction_length + 1):
        x = data[i:i + sequence_length]
        y = data[i + sequence_length:i + sequence_length + prediction_length]  # 预测接下来的5个时间步的值
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)
    def __getitem__(self, ix):
        if self.train:
            return self.X[ix], self.y[ix]
        return self.X[ix]

def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        for batch in dataloader:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds

if __name__ == '__main__':

    log_dir = './logs/homework2_RNN_5_day'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Deleted existing {log_dir} directory.")

    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "Summary of Weather.csv"

    weather_data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for item in reader:
            weather_data.append(float(item['MaxTemp']))

    # print(len(weather_data))
    # print(weather_data[:20])

    # 定义参数
    input_sequence_length = 30
    train_size = 20000
    test_size = 5000

    # 创建所有可能的序列
    all_x, all_y = create_sequences(weather_data, input_sequence_length)

    all_x = torch.tensor(all_x, dtype=torch.float32)
    all_y = torch.tensor(all_y, dtype=torch.float32)

    # 划分训练集和测试集
    X_train = all_x[:train_size].reshape(-1, input_sequence_length, 1)  # 添加通道维度
    y_train = all_y[:train_size].reshape(-1, 5)

    X_test = all_x[train_size:train_size+test_size].reshape(-1, input_sequence_length, 1)  # 添加通道维度
    y_test = all_y[train_size:train_size+test_size].reshape(-1, 5)

    print("X_train shape:", X_train.shape)  # torch.Size([20000, 30, 1])
    print("y_train shape:", y_train.shape)  # torch.Size([20000, 5])
    print("X_test shape:", X_test.shape)    # torch.Size([5000, 30, 1])
    print("y_test shape:", y_test.shape)    # torch.Size([5000, 5])

    dataset = {
        'train': TimeSeriesDataset(X_train, y_train),
        'test': TimeSeriesDataset(X_test, y_test, train=False)
    }

    dataloader = {
        'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
        'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
    }

    # 定义模型
    model = RNN_Classifier(input_size=1, hidden_size=20, num_layers=2, outpoust_size=5)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(dataloader['train']):
            X, y = X.to(device), y.to(device)
            # print(X.shape)
            # print(y.shape)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(dataloader['train']) + i)

    # 评估模型
    y_pred = predict(model, dataloader['test'])
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred.cpu()):.4f}")

