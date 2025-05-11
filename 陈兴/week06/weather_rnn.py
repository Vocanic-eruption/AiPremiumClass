import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# 2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#    数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2
# kaggle 上的训练结果:
# https://www.kaggle.com/code/mitrecx/notebook28b98ffd07

# 自动检测是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据
df = pd.read_csv("./陈兴/week06/data/weather.csv")

# 选择需要的列，例如日期和最高气温
df = df[["Date", "MaxTemp"]]

# 处理缺失值
df = df.dropna()

# 将日期转换为 datetime 格式
df["Date"] = pd.to_datetime(df["Date"])

# 按日期排序
df = df.sort_values(by="Date")

# 归一化最高气温
scaler = MinMaxScaler(feature_range=(0, 1))
df["MaxTemp"] = scaler.fit_transform(df[["MaxTemp"]])

# 创建序列
def create_sequences(data, seq_length, pred_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(sequences), np.array(labels)

seq_length = 30
pred_length = 5
sequences, labels = create_sequences(df["MaxTemp"].values, seq_length, pred_length)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

# 转换为 Tensor 并迁移到 device
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# 模型定义
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=5):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        # [num_layers, batch_size, hidden_size]
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)  # num_layers=2
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# 初始化模型与训练组件
model = WeatherRNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 日志
writer = SummaryWriter("./陈兴/week06/weather_runs/weather_prediction")

# 训练
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        writer.add_scalar("Loss/test", test_loss.item(), epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")

# 保存模型
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "./陈兴/week06/models/weather_rnn.pth")
writer.close()