import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

def load_weather_data(csv_path="week6_循环神经网络\\Summary of Weather.csv", target_col='MaxTemp', seq_len=30, pred_len=1):
    df = pd.read_csv(csv_path)
    df = df[[target_col]].dropna()
    data = df.values.flatten()

    # 构造滑动窗口数据
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])                   
        y.append(data[i+seq_len:i+seq_len+pred_len]) 

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        if self.train:
            return torch.from_numpy(self.X[ix]), torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])


class GRU(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size = 1,
            hidden_size = 50,
            bias = True,
            batch_first = True,
            num_layers = 2
        )
        self.fc = nn.Linear(50, output_size)
    def forward(self, x):
        outputs, _ = self.rnn(x)
        out = self.fc(outputs[:, -1, :])
        return out


def train_model(model, dataloader, log_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    writer = SummaryWriter(log_dir=log_dir)
    loop = tqdm(range(1, num_epochs + 1))
    for epoch in loop:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(-1)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        eval_loss = []
        with torch.no_grad():
            for X, y in dataloader['eval']:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(-1)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                eval_loss.append(loss.item())
        
        writer.add_scalar("Loss/train", np.mean(train_loss), epoch)
        writer.add_scalar("Loss/val", np.mean(eval_loss), epoch)
        loop.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")

    writer.close()
    return model
def predict(model,dataloader, pred_len=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    preds = torch.tensor([]).to(device)
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            X = X.unsqueeze(-1)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds.view(-1, pred_len).cpu().numpy()


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
            plt.autoscale() 
            if x_label and row == r - 1:
              plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
              plt.ylabel(y_label, fontsize=16, rotation=0)
    # plt.show()
    return fig

if __name__ == '__main__':

    csv_path = "week6_循环神经网络/Summary of Weather.csv"

    for pred_len, tag in zip([1, 5], ["one_day", "five_day"]):
        X, y = load_weather_data(csv_path, pred_len=pred_len)
        train_size = int(len(X) * 0.7)
        X_train, X_valid, X_test = X[:train_size], X[train_size:int(len(X)*0.9)], X[int(len(X)*0.9):]
        y_train, y_valid, y_test = y[:train_size], y[train_size:int(len(X)*0.9)], y[int(len(X)*0.9):]

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

        model = GRU(output_size=pred_len)
        model=train_model(model, dataloader, log_dir=f"runs/{tag}")

        preds = predict(model, dataloader['test'], pred_len=pred_len)
        fig = plot_series(X_test, y_test, preds, x_label="Time", y_label="Temperature")
        fig.savefig(f"runs/{tag}.png")