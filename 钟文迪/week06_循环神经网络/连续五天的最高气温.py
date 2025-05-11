import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

# 文件路径
file_path = './data/Summary of Weather.csv'

# 读取数据并生成时间序列
def generate_time_series(n_steps):
  with open(file_path, 'r', encoding='utf-8') as file:
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=str, encoding=None, invalid_raise=False)
    arr = data[:, 4].astype(float)  # 假设第4列是数值型数据
    time_series = [arr[i : i + n_steps] for i in range(len(arr) - n_steps)]  # 生成长度为50的时间序列
  return np.array(time_series, dtype=float)

n_steps = 50
series = generate_time_series(n_steps + 10)
series = np.expand_dims(series, axis=2)
X_train, y_train = series[:80000, :n_steps], series[:80000, -10:, 0]
X_valid, y_valid = series[80000:100000, :n_steps], series[80000:100000, -10:, -0]
X_test, y_test = series[100000:, :n_steps], series[100000:, -10:, 0]

# 数据集
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

# 循环神经网络模型
class RNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
    self.fc = torch.nn.Linear(20, 1)

  def forward(self, x):
    x, h = self.rnn(x) 
    y = self.fc(x[:,-1])
    return y

device = "cuda" if torch.cuda.is_available() else "cpu"

def fit(model, dataloader, epochs=10, log_dir="./钟文迪/week06_循环神经网络/runs/weather_prediction"):
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.MSELoss()
  writer = SummaryWriter(log_dir=log_dir)
  bar = tqdm(range(1, epochs+1))
  for epoch in bar:
    model.train()
    train_loss = []
    for batch in dataloader['train']:
      X, y = batch[0].float(), batch[1].float()
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()
      y_hat = model(X)
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      train_loss.append(loss.item())
    model.eval()
    eval_loss = []
    with torch.no_grad():
      for batch in dataloader['eval']:
        X, y = batch
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)
        eval_loss.append(loss.item())
    train_loss_mean = np.mean(train_loss)
    eval_loss_mean = np.mean(eval_loss)
    writer.add_scalar("five days Loss/Train", train_loss_mean, epoch)
    writer.add_scalar("five days Loss/Eval", eval_loss_mean, epoch)
    bar.set_description(f"loss {train_loss_mean:.5f} val_loss {eval_loss_mean:.5f}")
  writer.close()

def predict(model, dataloader):
  model.eval()
  with torch.no_grad():
    preds = torch.tensor([]).to(device)
    for batch in dataloader:
      X = batch.float()
      X = X.to(device)
      pred = model(X)
      preds = torch.cat([preds, pred])
    return preds

# 使用 num_layers参数来控制循环神经网络（RNN）中的隐藏层数量
class DeepRNN(torch.nn.Module):
  def __init__(self, n_in=50, n_out=1):
    super().__init__()
    self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=2, batch_first=True)
    self.fc = torch.nn.Linear(20, 1)

  def forward(self, x):
    x, h = self.rnn(x) 
    x = self.fc(x[:,-1])
    return x

rnn = DeepRNN()

# 训练模型并记录日志
fit(rnn, dataloader, epochs=10)

