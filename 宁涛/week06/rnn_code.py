import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=64,
                          hidden_size=100,
                          batch_first=True)
        self.f = nn.Linear(100, 40)

    def forward(self, x):
        outputs, l_h = self.rnn(x)
        out = self.f(outputs[:, -1, :])
        return out


def data_process():
    olivetti_faces = fetch_olivetti_faces(data_home="./data", shuffle=True, download_if_missing=True)
    images = torch.tensor(olivetti_faces.images)
    targets = torch.tensor(olivetti_faces.target)
    dataset = [(img, lbl) for img, lbl in zip(images, targets)]
    train_data, test_data = Data.random_split(dataset,[round(0.7*len(dataset)),round(0.3*len(dataset))])
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='logs')
    train_dataloader, test_dataloader = data_process()
    model = Model()
    model.to(device)
    epochs = 50
    loss_cn = nn.CrossEntropyLoss()
    accuracy = 0.0
    test_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            pre_lab = torch.argmax(output, dim=1)
            loss = loss_cn(output, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch：[{epoch + 1} / {epochs}], loss：{loss.item():.4f}')
                writer.add_scalar('rnn_train_loss', loss.item(), epoch * len(train_dataloader) + i)

        accuracy = 0.0
        with torch.no_grad():
            for X_t, y_t in test_dataloader:
                X_t, y_t = X_t.to(device), y_t.to(device)
                output = model(X_t)
                pre_lab = torch.argmax(output, dim=1)
                loss = loss_cn(output, y_t.long())
                accuracy += (output.argmax(1) == y_t).sum()

        print("整体测试集上的正确率：{}".format(accuracy / len(test_dataloader)))
        writer.add_scalar("rnn_test_accuracy", accuracy / len(test_dataloader), test_step)
        test_step += 1

    writer.close()

