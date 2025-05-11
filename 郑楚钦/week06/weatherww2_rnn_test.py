from torch import nn
from torch.utils.tensorboard import SummaryWriter
from rnnClassifier import RNN_Classifier
import torch
import csv
import numpy as np
from tqdm import tqdm
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=20, num_layers=1, dtype=torch.float64)
        self.fc = torch.nn.Linear(20, 5, dtype=torch.float64)

    def forward(self, x):
        x2, h = self.rnn(x)
        y = self.fc(h)
        return y


def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        last_y = None
        for batch in dataloader['train']:
            X, y, _ = batch
            if (last_y is not None):
                X = torch.tensor(y).to(device, dtype=torch.float64)[:, np.newaxis]
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, last_y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            last_y = y[np.newaxis, :]
        model.eval()
        eval_loss = []
        last_y = None
        with torch.no_grad():
            for batch in dataloader['eval']:
                dates, y, _ = batch
                if (last_y is not None):
                    X = torch.tensor(y).to(device, dtype=torch.float64)[:, np.newaxis]
                    y_hat = model(X)
                    loss = criterion(y_hat, last_y)
                    eval_loss.append(loss.item())
                    if epoch == epochs:
                        for i in range(len(dates)):
                            writer.add_scalars(key, {'y_hat': y_hat[0][i].item(), 'y': y[i].item()}, dates[i])                   
                last_y = y[np.newaxis, :]
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {
                            np.mean(eval_loss):.5f}")


def predict(key, model, data):
    writer = SummaryWriter("run")
    model.eval()
    with torch.no_grad():
        for X, y, date in data:
            pred = model(torch.tensor([[X]]).to(device, dtype=torch.float64))
            writer.add_scalars(
                key, {'y_hat': pred[0][0].item(), 'y': y}, X)
    writer.close()


with open('./week06/homework/Summary of Weather.csv', 'r') as f:
    writer = SummaryWriter("run")
    reader = csv.reader(f)
    print('表头', reader.__next__())
    staDict = {}
    beginDate = datetime.datetime.strptime('1940-01-01', '%Y-%m-%d')
    for row in reader:
        curDate = datetime.datetime.strptime(row[1], '%Y-%m-%d') - beginDate
        date = curDate.days
        # print(f'STA {row[0]}, Date {date}, MaxTemp {row[4]}')
        arr = staDict.setdefault(row[0], [])
        arr.append([date, float(row[4]), row[1]])
    batch_size = 5
    for key in staDict:
        data = staDict[key]
        length = len(data)
        print(f'key {key}, data length {length}')
        train_len = int(length * 0.8)
        data.sort(key=lambda x: x[0])
        train_dl = torch.utils.data.DataLoader(
            data[:train_len], batch_size=batch_size)
        val_dl = torch.utils.data.DataLoader(
            data[train_len:], batch_size=batch_size)
        rnn = RNN()
        fit(rnn, {'train': train_dl, 'eval': val_dl}, epochs=100)
        # y_pred = predict(key, rnn, data[train_len:])
    writer.close()
