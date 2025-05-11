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
    