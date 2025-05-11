import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

# 数据导入和预处理
df = pd.read_csv("Summary of Weather.csv", low_memory=False)
data = df[['STA', 'Date', 'Precip', 'MinTemp', 'MeanTemp', 'Snowfall', 'YR', 'MO', 'DA', 'PRCP', 'MaxTemp']].fillna(0)
data = data.replace(["T", '#VALUE!'], 0.0001)

# 添加滞后特征
data['MaxTemp_lag1'] = data['MaxTemp'].shift(1).fillna(0)

# 特征和目标
X = data[['STA', 'Precip', 'MinTemp', 'MeanTemp', 'Snowfall', 'YR', 'MO', 'DA', 'PRCP', 'MaxTemp_lag1']].values
y = data[['MaxTemp']].values
dates = data['Date'].values
stas = data['STA'].values

# 归一化特征
scaler = MinMaxScaler()
data_x = scaler.fit_transform(X)
data_y = scaler.fit_transform(y)

# 生成序列
def create_sequences(data_x, data_y, dates, stas, seq_length):
    x, y, date_seq, sta_seq = [], [], [], []
    for i in range(len(data_x) - seq_length):
        x_seq = data_x[i:i + seq_length]
        y_seq = data_y[i + seq_length]
        x.append(x_seq)
        y.append(y_seq)
        date_seq.append(dates[i + seq_length])
        sta_seq.append(stas[i + seq_length])
    return np.array(x), np.array(y), date_seq, sta_seq

seq_length = 5
x, y, date_seq, sta_seq = create_sequences(data_x, data_y, dates, stas, seq_length)

# 数据划分
xtrain, xtest, ytrain, ytest, datetrain, datetest, statrain, statest = train_test_split(
    x, y, date_seq, sta_seq, test_size=0.3, random_state=42
)
x_test, x_val, y_test, y_val, date_test, date_val, sta_test, sta_val = train_test_split(
    xtest, ytest, datetest, statest, test_size=0.5, random_state=42
)



# 转换为Tensor
xtrain = torch.tensor(xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# 数据加载器
batch_size = 64
train_dataset = TensorDataset(xtrain, ytrain)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型定义
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=100, batch_first=True)
        self.tanh = nn.Tanh
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = LSTM_Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
writer = SummaryWriter()
for epoch in range(epochs):
    model.train()
    for i,(inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
            print(f'{epoch+1}/{epochs},loss:{loss.item()}')


    model.eval()
    with torch.no_grad():
        for inputs,target in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs,target)
            Mse = np.mean((target.numpy() - outputs.numpy()) ** 2)
        writer.add_scalar('test mse', Mse, epoch)
        print(f'{epoch+1}/{epochs},mse:{Mse}')

writer.close()

# 预测函数
def predict_next_5_days(model, input_seq, scaler, seq_date, seq_sta, steps=5):
    predictions = []
    current_seq = input_seq.clone()

    for _ in range(steps):
        model.eval()
        with torch.no_grad():
            pred = model(current_seq)
            # 更新当前序列，移除最早的时间步，添加新的预测值
            current_seq = torch.roll(current_seq, -1, dims=1)
            current_seq[:, -1, -1] = pred.item()

        pred_unnorm = scaler.inverse_transform(pred.numpy())[0][0]
        predictions.append(pred_unnorm)

 # 假设最后一个特征是滞后特征

    return predictions, seq_date, seq_sta


# 测试预测
sample_idx = 0
sample_seq = x_test[sample_idx].unsqueeze(0)
sample_date = date_test[sample_idx]
sample_sta = sta_test[sample_idx]

y_pred_5days, start_date, start_sta = predict_next_5_days(
    model, sample_seq, scaler, sample_date, sample_sta
)

# 打印结果
for i, temp in enumerate(y_pred_5days):
    next_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
    print(f"STA: {start_sta}, Date: {next_date}, Predicted MaxTemp: {temp:.2f}")