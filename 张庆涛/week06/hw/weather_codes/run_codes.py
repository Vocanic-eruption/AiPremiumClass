import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,TensorDataset
from tqdm import tqdm
from Summary_of_Weather_model import SummaryOfWeater
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 数据加载
data = pd.read_csv('../data/Summary_of_Weather.csv')
# 数据预处理
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date') # 根据日期排序
data = data[['Date', 'MaxTemp']].dropna()  # 仅保留日期和最高气温列，并去除缺失值

# 归一化
scaler = MinMaxScaler()
data['MaxTemp'] = scaler.fit_transform(data[['MaxTemp']])

def create_sequences(data, seq_length):
  """
    将时间序列数据转换为模型输入格式
    : data  包含世界序列数据的 dataframe
    : seq_length  每个序列的长度，也就是时间序列的尝试 过去的多少天数据
    : return  输入数据x和目标值以
  """
  xs,ys = [],[] 
  for i in range(len(data)-seq_length):
    x = data.iloc[i:i+seq_length]['MaxTemp'].values
    y = data.iloc[i+seq_length]['MaxTemp']
    xs.append(x)
    ys.append(y)
  return np.array(xs),np.array(ys)
# 创建序列
seq_length = 30 # 使用过去三十天的数据
X,y = create_sequences(data,seq_length) # 构建时间序列数据

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 进一步划分训练集和验证集
train_size = int(0.8 * len(train_dataset))  # 80%训练集
valid_size = len(train_dataset) - train_size  # 20%验证集
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32)
# 创建 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
      
#       
dataset = WeatherDataset(X, y)

# 划分训练集、验证集、测试集
# 划分训练集，验证集、测试集
train_size = int(0.6 * len(X))
valid_size = int(0.2 * len(X))
test_size = len(X) - train_size - valid_size

train_data, test_data,valid_data = torch.utils.data.random_split(dataset, [train_size, test_size,valid_size])

# 创建 dataLoader
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
# 模型训练 创建模型
input_size=1
hidden_size=64
output_size=1
model = SummaryOfWeater(input_size, hidden_size, output_size,num_layers=2)

# 损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 设置 tensorBoard
writer = SummaryWriter()
# 训练模型
epochs = 10
train_losses = []
valid_losses =[]
test_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.unsqueeze(1)  # 调整目标值的形状
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    # 记录训练损失
    train_losses.append(train_loss/len(train_loader))
    # 在 tensorboard 中记录训练损失
    writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
    # 验证集
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader):
            outputs = model(inputs)
            targets = targets.unsqueeze(1)  # 调整目标值的形状
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
    # 记录验证损失
    valid_losses.append(valid_loss/len(valid_loader))
    # 在 tensorboard 中记录验证损失
    writer.add_scalar('Loss/valid', valid_loss/len(valid_loader), epoch)
    
      # 测试集
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            outputs = model(inputs)
            targets = targets.unsqueeze(1)  # 调整目标值的形状
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    # 记录测试损失
    test_losses.append(test_loss/len(test_loader))
    # 在 tensorboard 中记录测试损失
    writer.add_scalar('Loss/test', test_loss/len(test_loader), epoch)
writer.close()
# 绘制可视化图形
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.grid()
plt.legend()
plt.show()

# 预测一天的1天的最高气温
model.eval()
with torch.no_grad():
    one_day_pred = model(X_test[-1].unsqueeze(0)) # 取最后一个样本
    one_day_pred = scaler.inverse_transform(one_day_pred.numpy())
    
# 预测连续五天的最高气温
five_day_preds = []
last_seq = X_test[-1]
for _ in range(5):
    with torch.no_grad():
        pred = model(last_seq.unsqueeze(0))
        five_day_preds.append(pred.item())
        last_seq = torch.cat([last_seq[1:], pred.view(1, 1)], dim=0)  # 确保维度一致
five_day_preds = scaler.inverse_transform(np.array(five_day_preds).reshape(-1, 1))  # 反归一化

print("预测1天的最高气温:", one_day_pred)
print("预测连续5天的最高气温:", five_day_preds)


# 绘制预测结果
plt.figure(figsize=(10, 5))
plt.plot(range(len(five_day_preds)), five_day_preds, marker='o', label='Predicted Max Temperature')
plt.xlabel('Day')
plt.ylabel('Max Temperature')
plt.title('Predicted Max Temperature for Next 5 Days')
plt.legend()
plt.grid()
plt.show()