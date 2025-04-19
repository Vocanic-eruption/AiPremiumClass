import os
import tqdm

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def show_picture(path):
    # 读取图片
    img = plt.imread(path)  # 返回 NumPy 数组

    # 显示图片
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


def iqr(data, threshold=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    data = data[(data >= lower_bound) & (data <= upper_bound)]
    return data


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


class WhetherPredict(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layer,
                 num_classes):
        super().__init__()
        self.rnn = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
        )
        # nn layer
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes,
        )

    def forward(self, X):
        output, h_n = self.rnn(X)
        y = self.classifier(output[:, -1])
        return y


def sliding_window(array, window_size):
    array_list = sliding_window_view(array, window_shape=window_size, step_size=1)
    return array_list


def generate_samples(data, length=200, batch=1000, features=1, predict_n=1):
    batch_data = np.empty([batch, length+predict_n, features], dtype='float32')
    samples = sliding_window_view(data.values[:, :features],
                                  window_shape=length+predict_n,
                                  axis=0)
    samples = np.transpose(samples, (0, 2, 1))
    select_cols = np.random.choice(samples.shape[0],
                                          batch,
                                          replace=False
                                        )
    batch_data[:, :, :] = samples[select_cols]
    train_data = batch_data[:, :length, :]
    label_data = batch_data[:, -predict_n:, 0]
    return train_data, label_data


def load_data(data_path, min_size):
    location_data = pd.read_csv(
        os.path.join(data_path, 'Weather Station Locations.csv'))
    weather_data = (pd.read_csv(
        os.path.join(data_path, 'Summary of Weather.csv'))
                    .set_index('Date'))
    data = weather_data.join(location_data.set_index('WBAN'), how='left', on='STA')
    print(weather_data.head())
    states = weather_data['STA'].unique()
    states_dict = {}
    for state in states:
        state_weather = data[data['STA'] == state][
            ['MaxTemp',
            'Longitude',
            'Latitude']
        ]
        clean_data = iqr(state_weather['MaxTemp'])
        states_dict.update({state: state_weather.loc[clean_data.index]})
    data1 = [x.shape[0] for x in states_dict.values()]
    n, bins, patches = plt.hist(
        x=data1,  # 输入数据
        bins=30,  # 柱子数量或边界数组
        density=True,  # 显示概率密度(代替频次)
        color='#66b3ff',  # 柱体颜色
        edgecolor='black',  # 边框颜色
        alpha=0.7,  # 透明度
        label='正态分布',
        histtype='bar'  # 类型：默认柱状
    )
    # plt.show()
    states_dict = {k: v for k, v in states_dict.items() if v.shape[0] > min_size}
    return states_dict

def plot_series(data, label,  y_pred=None, predict_n=None):
    r, c = 2, 2
    plot_data = data[:r*c, :, 0]
    label = label[:r*c, :]
    fig, axes = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            x = list(range(plot_data.shape[1] + predict_n))
            if predict_n == 1:
                y = plot_data[cnt].tolist() + [label[cnt]]
            else:
                y = plot_data[cnt].tolist() + label[cnt].tolist()
            if y_pred is not None:
                axes[i, j].plot(x[:-predict_n], y[:-predict_n])
                axes[i, j].plot(x[-predict_n:], y[-predict_n:],
                                   'rv', color='b', label='y')
                axes[i, j].scatter(x[-predict_n:], y_pred[cnt, :],
                                   marker='x', color='r', label='y_pred')
            else:
                axes[i, j].plot(x, y, linestyle="-.")
            axes[i, j].set_xlabel('data')
            axes[i, j].set_ylabel('temperature')
            axes[i, j].legend(loc='best')
            cnt += 1
    plt.tight_layout()
    plt.show()


def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    bar = tqdm.tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar('Temperature Training Loss (per epoch)', loss.item(), epoch)
        model.eval()
        eval_loss = []
        with torch.no_grad():
            for batch in dataloader['test']:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                eval_loss.append(loss.item())
                writer.add_scalar('Temperature Val Loss (per epoch)', loss.item(), epoch)
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")


def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds



if __name__ == '__main__':
    # pic_path = r'runs/loss.png'
    # show_picture(pic_path)
    weather_data_path = r'../data/archive'
    device = "cpu"
    train_data_length=30
    predict_n = 1  # 预测点数长度
    features = 1 # 特征个数
    batch=1000
    epochs = 500

    # 提取气温数据，造样本
    states_whether_data = load_data(weather_data_path, min_size=batch + train_data_length + predict_n)
    state = np.random.choice(list(states_whether_data.keys()), size=1)
    print(f'select state: {state}')
    # 目前生成一个州的数据

    data, label = generate_samples(states_whether_data[state[0]],
                                  length=train_data_length,
                                  batch=batch,
                                  features=features,
                                  predict_n=predict_n)
    # 画曲线图
    # plot_series(data, label, y_pred=label, predict_n=predict_n)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    dataset = {
        'train': TimeSeriesDataset(X_train, y_train),
        # 'eval': TimeSeriesDataset(X_valid, y_valid),
        'test': TimeSeriesDataset(X_test, y_test)
    }
    dataloader = {
        'train': DataLoader(dataset['train'], shuffle=True, batch_size=20),
        # 'eval': DataLoader(dataset['eval'], shuffle=True, batch_size=20),
        'test': DataLoader(dataset['test'], shuffle=True, batch_size=20),
    }

    model = WhetherPredict(
        input_size=features,
        hidden_size=50,
        num_layer=1,
        num_classes=predict_n)
    writer = SummaryWriter(f'runs/lstm_pred{predict_n}')  # 日志会保存在 runs/experiment_1 目录
    # 单点预测
    fit(model, dataloader, epochs=epochs)
    preds = predict(model, dataloader['test'])
    plot_series(X_test, y_test, y_pred=preds, predict_n=predict_n)
    # 多点预测



