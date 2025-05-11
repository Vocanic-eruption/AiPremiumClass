import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 加载数据集
url = 'https://www.kaggle.com/datasets/smid80/weatherww2'
# 数据集已经下载并解压到本地
data = pd.read_csv('Summary of Weather.csv')

# 数据预处理
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.dropna(subset=['MaxTemp'], inplace=True)
scaler = MinMaxScaler()
data['MaxTemp'] = scaler.fit_transform(data['MaxTemp'].values.reshape(-1, 1))

# 创建时间序列数据
def create_sequences(data, seq_length):
    # 创建两个空列表，用于存储输入序列和输出序列
    xs, ys = [], []
    # 遍历数据，从第一个元素开始，到倒数第seq_length个元素结束
    for i in range(len(data) - seq_length):
        # 取出从第i个元素开始的seq_length个元素作为输入序列
        x = data[i:i+seq_length]
        # 取出第i+seq_length个元素作为输出序列
        y = data[i+seq_length]
        # 将输入序列和输出序列分别添加到对应的列表中
        xs.append(x)
        ys.append(y)
    # 将列表转换为numpy数组并返回
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(data['MaxTemp'].values, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义RNN模型
class WeatherRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        # 初始化函数，传入输入大小、隐藏层大小、层数和输出大小
        super(WeatherRNNModel, self).__init__()
        # 调用父类的初始化函数
        self.hidden_size = hidden_size
        # 将隐藏层大小赋值给实例变量
        self.num_layers = num_layers
        # 将层数赋值给实例变量
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 初始化RNN层，传入输入大小、隐藏层大小、层数和batch_first参数
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态h0，大小为(num_layers, x.size(0), self.hidden_size)，并将x的设备赋值给h0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 将x和h0传入rnn，得到输出out和隐藏状态h
        out, _ = self.rnn(x, h0)
        # 将out的最后一个时间步的输出传入全连接层fc，得到最终输出
        out = self.fc(out[:, -1, :])
        # 返回最终输出
        return out

# 训练函数
def train_model(model, train_loader, criterion, optimizer, writer, num_epochs=25):
    # 将模型设置为训练模式
    model.train()
    # 遍历每个epoch
    for epoch in range(num_epochs):
        # 初始化每个epoch的损失
        running_loss = 0.0
        # 遍历每个batch
        for inputs, targets in train_loader:
            # 将梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

        # 计算每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        # 将损失写入tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # 打印每个epoch的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 测试函数
def test_model(model, test_loader, criterion, writer):
    # 将模型设置为评估模式
    model.eval()
    # 初始化损失值为0
    running_loss = 0.0
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据集
        for inputs, targets in test_loader:
            # 使用模型进行预测
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 累加损失
            running_loss += loss.item()

    # 计算平均损失
    epoch_loss = running_loss / len(test_loader)
    # 将损失值写入tensorboard
    writer.add_scalar('Loss/test', epoch_loss, 0)
    # 打印损失值
    print(f'Test Loss: {epoch_loss:.4f}')

# 主函数
def main():
    # 定义输入层大小
    input_size = 1
    # 定义隐藏层大小
    hidden_size = 128
    # 定义层数
    num_layers = 2
    # 定义输出层大小
    output_size = 1
    # 定义训练轮数
    num_epochs = 25
    # 定义学习率
    learning_rate = 0.001

    # 创建模型
    model = WeatherRNNModel(input_size, hidden_size, num_layers, output_size)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 创建SummaryWriter对象，用于记录训练过程中的数据
    writer = SummaryWriter('runs/weather_rnn')
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, writer, num_epochs)
    # 测试模型
    test_model(model, test_loader, criterion, writer)
    # 关闭SummaryWriter对象
    writer.close()

if __name__ == '__main__':
    main()