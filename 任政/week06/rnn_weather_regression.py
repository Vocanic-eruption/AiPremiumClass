import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 加载数据集
def load_data(url):
    # 从 URL 加载数据集
    data = pd.read_csv(url , low_memory = False)
    # 选择最高气温列作为目标值
    max_temp = data['MaxTemp'].values
    # 将数据转换为 PyTorch 张量
    max_temp = torch.tensor(max_temp, dtype=torch.float32)
    return max_temp

# 数据预处理
def preprocess_data(max_temp, seq_length):
    X, y = [], []
    # 将数据划分为序列
    for i in range(len(max_temp) - seq_length):
        X.append(max_temp[i:i+seq_length])  # 输入序列
        y.append(max_temp[i+1:i+seq_length+1])  # 目标序列
    # 转换为 PyTorch 张量
    X = torch.stack(X)
    y = torch.stack(y)
    return X, y

# 定义 RNN 模型
class RNN_Weather_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN_Weather_Predictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN 前向传播
        out, _ = self.rnn(x)
        # 使用最后一个时间步的输出进行预测
        out = self.fc(out[:, -1, :])
        return out

# 主函数
if __name__ == '__main__':
    # 数据集 URL
    LOCAL_PATH = './data/weather_data/Summary of Weather.csv'
    # 加载数据
    max_temp = load_data(LOCAL_PATH)
    # 数据预处理
    seq_length = 10  # 输入序列长度
    X, y = preprocess_data(max_temp, seq_length)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 转换为 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 实例化模型
    input_size = 1  # 输入特征维度
    hidden_size = 50  # 隐藏层神经元数量
    output_size = 1  # 输出特征维度
    num_layers = 2  # RNN 层数
    model = RNN_Weather_Predictor(input_size, hidden_size, output_size, num_layers)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 实例化 TensorBoard
    writer = SummaryWriter('runs/weather_prediction')
    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # 前向传播
            outputs = model(inputs.unsqueeze(-1))  # 添加特征维度
            # 计算损失
            loss = criterion(outputs, targets[:, -1].unsqueeze(-1))  # 只预测最后一个时间步
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录损失到 TensorBoard
            if i % 100 == 0:
                writer.add_scalar(f'Training Loss', loss.item(), epoch * len(train_loader) + i)
        # 评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for  i, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs.unsqueeze(-1))
                test_loss += criterion(outputs, targets[:, -1].unsqueeze(-1)).item()
            test_loss /= len(test_loader)
            if i %100 == 0:
                writer.add_scalar(f'Test Loss', test_loss, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')
    # 关闭 TensorBoard
    writer.close()
