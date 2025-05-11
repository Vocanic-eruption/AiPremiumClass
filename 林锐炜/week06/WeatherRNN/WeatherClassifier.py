import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 配置参数
class Config:
    # 数据路径
    data_path = "/Users/linjavie/myzone/code/python/python_basic_learning/AI/pytorch_note/PyTorch_RNN/WeatherRNN/Summary of Weather.csv" 
    # 选择需要的特征列
    feature_columns = ['MaxTemp', 'MinTemp', 'Precip', 'Snowfall', 'PoorWeather']
    target_column = 'MaxTemp'  # 预测目标
    sequence_length = 30       # 输入序列长度（使用30天预测未来）
    
    # 模型参数
    input_size = 5            # 输入特征维度
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    output_size = 5           # 预测未来5天的最高气温
    
    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    train_ratio = 0.8
    val_ratio = 0.1
    
    # TensorBoard保存路径
    log_dir = "runs/weather_rnn"
    model_save_path = "models/rnn_weather.pth"

# 数据预处理类
class WeatherDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def load_data(self):
    # 加载数据时处理混合类型问题
        df = pd.read_csv(self.config.data_path, 
                        parse_dates=['Date'],
                        low_memory=False)
        
        # 确保按日期排序
        df = df.sort_values('Date')
        
        # 处理PoorWeather列
        if 'PoorWeather' in df.columns:
            # 替换所有可能的坏天气标记为1
            poor_weather_values = ['1', '1.0', '1,', '1.', '1]', '[1']
            df['PoorWeather'] = df['PoorWeather'].fillna('0').astype(str).str.extract(r'(\d)')[0].fillna('0').astype(int)
        
        # 填充其他特征列的缺失值
        for col in self.config.feature_columns:
            if col in df.columns and col != 'PoorWeather':
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].ffill().fillna(0)
                else:
                    df[col] = df[col].ffill().fillna('0')
        
        # 转换特征列为数值类型
        features = df[self.config.feature_columns].apply(pd.to_numeric, errors='coerce')
        targets = df[[self.config.target_column]].apply(pd.to_numeric, errors='coerce')
        
        # 标准化前填充可能出现的NaN（由于强制转换产生的）
        features = features.fillna(0)
        targets = targets.fillna(0)
        
        # 使用同一个scaler进行标准化（保证特征和目标的相对关系）
        all_data = pd.concat([features, targets], axis=1)
        scaled_data = self.scaler.fit_transform(all_data)
        
        # 分离特征和目标
        scaled_features = scaled_data[:, :-1]
        scaled_targets = scaled_data[:, -1].reshape(-1, 1)
        
        return scaled_features, scaled_targets


    def create_sequences(self, data, targets):
        xs, ys = [], []
        for i in range(len(data) - self.config.sequence_length - self.config.output_size):
            x = data[i:i+self.config.sequence_length]
            y = targets[i+self.config.sequence_length:i+self.config.sequence_length+self.config.output_size]
            xs.append(x)
            ys.append(y.flatten())  # 展平为5天的预测
        return np.array(xs), np.array(ys)

# 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features).to(device)
        self.targets = torch.FloatTensor(targets).to(device)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# RNN模型
class WeatherRNN(nn.Module):
    def __init__(self, config):
        super(WeatherRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        ).to(device)
        self.fc = nn.Linear(config.hidden_size, config.output_size).to(device)
        
    def forward(self, x):
        x = x.to(device)
        out, (h_n, c_n) = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])    # 使用最后一个时间步的输出
        return out

# 训练函数
def train(config):
    # 创建日志目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(config.log_dir)
    
    # 数据处理
    processor = WeatherDataProcessor(config)
    features, targets = processor.load_data()
    X, y = processor.create_sequences(features, targets)
    
    # 数据集划分
    total_samples = len(X)
    train_size = int(total_samples * config.train_ratio)
    val_size = int(total_samples * config.val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # 创建DataLoader
    train_loader = DataLoader(
        WeatherDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        WeatherDataset(X_val, y_val),
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        WeatherDataset(X_test, y_test),
        batch_size=config.batch_size
    )
    
    # 初始化模型
    model = WeatherRNN(config).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for i,(batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录TensorBoard日志
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
        
        print(f'Epoch [{epoch+1}/{config.epochs}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f}')
    
    # 测试阶段
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
            avg_test_loss = test_loss / len(test_loader)
            print(f'Final Test Loss: {avg_test_loss:.4f}')
            writer.add_scalar('Loss/test', avg_test_loss)
    writer.close()

if __name__ == "__main__":
    config = Config()
    train(config)
