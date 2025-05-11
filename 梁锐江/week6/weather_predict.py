import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class WhetherRnnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,
            hidden_size=30,
            num_layers=2,
            batch_first=True
        )
        self.fc = torch.nn.Linear(30, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        return self.fc(outputs[:, -1, :])


def convert_and_split(data, look_back_steps, predict_steps):
    X, y = [], []
    for i in range(len(data) - look_back_steps - predict_steps):
        X.append(data[i:i + look_back_steps])
        y.append(data[i + look_back_steps: i + look_back_steps + predict_steps])
    return np.array(X), np.array(y)


def create_datasets(look_back_steps, predict_steps):
    weather_csv = './data/Summary of Weather.csv'
    data = pd.read_csv(weather_csv, low_memory=False, parse_dates=['Date'])
    data = data[['Date', 'MaxTemp']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['MaxTemp'] = scaler.fit_transform(data['MaxTemp'].values.reshape(-1, 1))
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    X, y = convert_and_split(data, look_back_steps, predict_steps)
    split_len = int(len(X) * 0.8)

    X_train = torch.tensor(X[:split_len], dtype=torch.float32)
    y_train = torch.tensor(y[:split_len], dtype=torch.float32)
    X_test = torch.tensor(X[split_len:], dtype=torch.float32)
    y_test = torch.tensor(y[split_len:], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def train_model(train_dl, test_dl, predict_steps, epochs=100):
    writer = SummaryWriter("./runs/weather_predict" + str(predict_steps))
    model = WhetherRnnModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    step = 0

    bar = tqdm(range(epochs))
    for epoch in bar:
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_dl:
            params, result = X_batch.to(device), y_batch.to(device)
            optim.zero_grad()
            pred = model(params)
            loss = loss_func(pred, result)
            epoch_loss += loss.item()
            loss.backward()
            optim.step()
            step += 1

        avg_loss = epoch_loss / len(train_dl)
        writer.add_scalar("Loss", avg_loss, epoch)

        # 评估
        model.eval()

        test_epoch_loss = 0
        for test_batch_x, test_batch_y in test_dl:
            test_batch_x, test_batch_y = test_batch_x.to(device), test_batch_y.to(device)
            pred = model(test_batch_x)
            loss = loss_func(pred, test_batch_y)
            test_epoch_loss += loss.item()
        test_avg_epoch_loss = test_epoch_loss / len(test_dl)
        writer.add_scalar("Test Loss", test_avg_epoch_loss, epoch)

    writer.close()


def train_with_feature_size(predict_steps):
    print(f"正在训练预测{predict_steps}天模型")
    train_batch, test_batch = create_datasets(look_back_steps=7, predict_steps=1)
    train_model(train_batch, test_batch, predict_steps)


if __name__ == '__main__':
    train_with_feature_size(1)
    train_with_feature_size(5)
