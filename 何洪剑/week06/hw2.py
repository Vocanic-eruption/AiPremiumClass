import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 数据预处理
class WeatherDataset(Dataset):
    def __init__(self, data, seq_length, pred_days):
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data[['MaxTemp']])
        
        self.sequences = []
        self.labels = []
        
        for i in range(len(scaled_data) - seq_length - pred_days):
            self.sequences.append(scaled_data[i:i+seq_length])
            self.labels.append(scaled_data[i+seq_length:i+seq_length+pred_days])
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor(self.labels[idx])
        return sequence, label

# 2. 模型定义
class WeatherLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_days=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, output_days)
        
    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out.unsqueeze(-1)  # 保持输出维度一致

# 3. 训练函数
def train_model(model_type='1day'):
    # 配置参数
    SEQ_LENGTH = 30
    PRED_DAYS = 1 if model_type == '1day' else 5
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # 创建日志目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(current_dir, 'runs/weather')
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(runs_dir, model_type))
    
    # 加载数据
    data = pd.read_csv(os.path.join(current_dir,'Summary of Weather.csv'), parse_dates=['Date'])
    data = data.dropna(subset=['MaxTemp']).sort_values('Date')
    
    # 数据集划分
    train_size = int(0.8 * len(data))
    train_data = WeatherDataset(data[:train_size], SEQ_LENGTH, PRED_DAYS)
    test_data = WeatherDataset(data[train_size:], SEQ_LENGTH, PRED_DAYS)
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLSTM(output_days=PRED_DAYS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    best_test_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 记录训练损失
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # 验证过程
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                test_loss += criterion(outputs, labels).item()
        
        avg_test_loss = test_loss / len(test_loader)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), f'best_{model_type}_model.pth')
    
    writer.close()
    return model

# 4. 主程序
if __name__ == '__main__':
    # 训练1天预测模型
    train_model(model_type='1day')
    
    # 训练5天预测模型 
    train_model(model_type='5days')