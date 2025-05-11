import kagglehub
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("smid80/weatherww2")

print("Path to dataset files:", path)

# data = pd.read_csv(f"{path}/Summary of Weather.csv")
# print(data.shape)

# 加载数据
df = pd.read_csv(os.path.join(path, "Summary of Weather.csv"))
df = df[['Date', 'MaxTemp']].dropna()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 数据预处理
scaler = MinMaxScaler()
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size+val_size]
test_data = df.iloc[train_size+val_size:]

scaler.fit(train_data[['MaxTemp']])
train_tmax = scaler.transform(train_data[['MaxTemp']]).flatten()
val_tmax = scaler.transform(val_data[['MaxTemp']]).flatten()
test_tmax = scaler.transform(test_data[['MaxTemp']]).flatten()

# 创建数据集
class WeatherDataset(Dataset):
    def __init__(self, data, seq_len=7, pred_len=5):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(x).float().unsqueeze(1), torch.tensor(y).float()

train_dataset = WeatherDataset(train_tmax)
val_dataset = WeatherDataset(val_tmax)
test_dataset = WeatherDataset(test_tmax)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义模型
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WeatherRNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TensorBoard
writer = SummaryWriter('runs/weather_prediction')

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    # 记录损失
    writer.add_scalars('Loss', {
        'train': train_loss/len(train_loader),
        'val': val_loss/len(val_loader)
    }, epoch)

# 测试和可视化
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        preds = model(inputs).cpu().numpy()
        predictions.extend(preds)
        actuals.extend(targets.numpy())

predictions = scaler.inverse_transform(np.array(predictions))
actuals = scaler.inverse_transform(np.array(actuals))

# 保存预测结果图像
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.plot(actuals[:, i], label='Actual')
    plt.plot(predictions[:, i], label='Predicted')
    plt.title(f'Day {i+1}')
    plt.legend()
plt.tight_layout()
plt.savefig('predictions.png')

# 保存模型
torch.save(model.state_dict(), 'weather_model.pth')
writer.close()