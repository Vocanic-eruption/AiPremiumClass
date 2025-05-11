import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,   ）
            hidden_size=128, 
            num_layers=3,     
            batch_first=True  
        )
        self.fc = nn.Linear(128, 40) 

    def forward(self, x):
        # 输入 x 形状: [batch, time_steps, features]
        outputs, _ = self.rnn(x)
        out = self.fc(outputs[:, -1, :])  # 取最后时间步的输出
        return out


if __name__ == '__main__':

    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    faces = fetch_olivetti_faces(shuffle=True)
    X, y = faces.images, faces.target

    X = X.reshape(-1, 64, 64).astype('float32')  
    y = torch.tensor(y, dtype=torch.long)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test), y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型
    model = RNN_Classifier().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], 64, 64)  # 调整为 (batch, time_steps, features)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('average training loss', avg_loss, epoch)

        # 评估模型
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], 64, 64)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('test accuracy', accuracy, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # 记录模型结构到 TensorBoard
    sample_input = torch.zeros(1, 64, 64).to(device)  # 1 张人脸数据
    writer.add_graph(model, sample_input)

    # 保存模型
    # torch.save(model, 'rnn_model.pth')
    torch.save(model.state_dict(), 'rnn_model_params.pth')

    writer.close()

    # 加载模型
    # model = torch.load('rnn_model.pth')

    # 加载模型参数
    model = RNN_Classifier()
    model.load_state_dict(torch.load('rnn_model_params.pth'))
    model.to(device)
 


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('C:/Users/Elysia/Desktop/Summary of Weather.csv/Summary of Weather.csv') 
max_temp = data['MaxTemp'].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler()
scaled_max_temp = scaler.fit_transform(max_temp)

def plot_series(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$"):
    r, c = 3, 5
    fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))
    for row in range(r):
        for col in range(c):
            plt.sca(axes[row][col])
            ix = col + row*c
            plt.plot(series[ix, :], ".-")
            if y is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y[ix])), y[ix], "bx", markersize=10)
            if y_pred is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix], "ro")
            if y_pred_std is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] + y_pred_std[ix])
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] - y_pred_std[ix])
            plt.grid(True)
            plt.hlines(0, 0, 100, linewidth=1)
            plt.axis([0, len(series[ix, :])+len(y[ix]), -1, 1])
            if x_label and row == r - 1:
              plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
              plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.show()

# 准备时间序列数据
def create_sequences(data, n_steps, pred_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - pred_steps + 1):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps:i+n_steps+pred_steps])
    return np.array(X), np.array(y)

n_steps = 30  # 时间步长
pred_steps = 3  # 预测天数
X, y = create_sequences(scaled_max_temp, n_steps, pred_steps)

# 划分训练集和测试集
train_size = int(len(X) * 0.7)
valid_size = int(len(X) * 0.2)
X_train, X_valid, X_test = X[:train_size], X[train_size:train_size + valid_size], X[train_size + valid_size:]
y_train, y_valid, y_test = y[:train_size], y[train_size:train_size + valid_size], y[train_size + valid_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        if self.train:
            return torch.from_numpy(self.X[ix]).float(), torch.from_numpy(self.y[ix]).float()
        return torch.from_numpy(self.X[ix]).float()

dataset = {
    'train': TimeSeriesDataset(X_train, y_train),
    'eval': TimeSeriesDataset(X_valid, y_valid),
    'test': TimeSeriesDataset(X_test, y_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        # 修改全连接层输出维度为预测天数
        self.fc = torch.nn.Linear(20, pred_steps)

    def forward(self, x):
        x, h = self.rnn(x) 
        # 取最后时间步的输出
        y = self.fc(x[:, -1])
        return y

rnn = RNN()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 配置 TensorBoard
writer = SummaryWriter()

def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            # Reshape y to match the shape of y_hat if necessary
            y = y.squeeze(-1)  # Remove the last dimension if it's singleton
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_avg_loss = np.mean(train_loss)

        model.eval()
        eval_loss = []
        with torch.no_grad():
            for batch in dataloader['eval']:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                y = y.squeeze(-1)  # Remove the last dimension if it's singleton
                loss = criterion(y_hat, y)
                eval_loss.append(loss.item())
        eval_avg_loss = np.mean(eval_loss)

        bar.set_description(f"loss {train_avg_loss:.5f} val_loss {eval_avg_loss:.5f}")

        # Record loss to TensorBoard
        writer.add_scalar('Training Loss', train_avg_loss, epoch)
        writer.add_scalar('Validation Loss', eval_avg_loss, epoch)

    writer.close()

def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = []
        for batch in dataloader:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        return preds

fit(rnn, dataloader)

y_pred = predict(rnn, dataloader['test'])
# y_pred = y_pred.cpu().numpy()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, pred_steps)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, pred_steps)


# 评估模型，这里取所有预测天数的平均 MSE
mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
print(f'Mean Squared Error: {mse}')

plt.plot(np.arange(n_steps), scaler.inverse_transform(X_test[0]).flatten(), label='Historical Data')
plt.plot(np.arange(n_steps, n_steps + pred_steps), y_test[0], label='Actual')
plt.plot(np.arange(n_steps, n_steps + pred_steps), y_pred[0], label='Predicted')
plt.legend()
plt.show()
