import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

LR = 1e-2
EPOCHS = 800
BATCH_SIZE = 32


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(in_features=4096, out_features=128)
        self.sg = nn.Sigmoid()
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.rl = nn.ReLU()
        self.ln3 = nn.Linear(in_features=64, out_features=40)

    def forward(self, x):
        out = self.ln(x)
        out = self.sg(out)
        out = self.ln2(out)
        out = self.rl(out)
        final = self.ln3(out)

        return final


class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(in_features=4096, out_features=128)
        self.bn = nn.BatchNorm1d(128)
        self.sg = nn.Sigmoid()
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.rl = nn.ReLU()
        self.ln3 = nn.Linear(in_features=64, out_features=40)

    def forward(self, x):
        out = self.ln(x)
        out = self.bn(out)
        out = self.sg(out)
        out = self.ln2(out)
        out = self.rl(out)
        final = self.ln3(out)
        return final


class ModelWithDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(in_features=4096, out_features=128)
        self.sg = nn.Sigmoid()
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.dp = nn.Dropout(0.2)
        self.rl = nn.ReLU()
        self.ln3 = nn.Linear(in_features=64, out_features=40)

    def forward(self, x):
        out = self.ln(x)
        out = self.sg(out)
        out = self.ln2(out)
        out = self.rl(out)
        out = self.dp(out)
        final = self.ln3(out)
        return final


class ModelWithBNAndDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(in_features=4096, out_features=128)
        self.bn = nn.BatchNorm1d(128)
        self.sg = nn.Sigmoid()
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.dp = nn.Dropout(0.2)
        self.rl = nn.ReLU()
        self.ln3 = nn.Linear(in_features=64, out_features=40)

    def forward(self, x):
        out = self.ln(x)
        out = self.sg(out)
        out = self.ln2(out)
        out = self.rl(out)
        out = self.dp(out)
        final = self.ln3(out)
        return final


class ModelWithDPAndBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(in_features=4096, out_features=128)
        self.sg = nn.Sigmoid()
        self.dp = nn.Dropout(0.2)
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.bn = nn.BatchNorm1d(128)
        self.rl = nn.ReLU()
        self.ln3 = nn.Linear(in_features=64, out_features=40)

    def forward(self, x):
        out = self.ln(x)
        out = self.sg(out)
        out = self.ln2(out)
        out = self.rl(out)
        out = self.dp(out)
        final = self.ln3(out)
        return final


def draw_train_hist(hist):
    plt.plot(hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Training Loss')
    plt.show()


def train_model(epochs, model, train_dl, loss_fn, optimizer):
    history = []
    for epoch in range(epochs):
        for X, y in train_dl:
            predict = model(X)
            # 计算损失
            loss = loss_fn(predict, y)
            # 计算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(loss.item())
    return history


def train_model_v2(epochs, model, train_dl, loss_fn, optimizer, device):
    history = []
    for epoch in range(epochs):
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            predict = model(X)
            # 计算损失
            loss = loss_fn(predict, y)
            # 计算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(loss.item())
    return history


def train_model_without_optim():
    model = SimpleModel()
    olivetti_faces = fetch_olivetti_faces(data_home="F:/githubProject/AiPremiumClass/梁锐江/week4", shuffle=True)
    train_data = TensorDataset(torch.tensor(olivetti_faces.data), torch.tensor(olivetti_faces.target, dtype=torch.long))
    # print(len(train_data))
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    for epoch in range(EPOCHS):
        for X, y in dataloader:
            predict = model(X)
            # 计算损失
            loss = loss_fn(predict, y)
            # 计算梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append(loss.item())
    draw_train_hist(history)


if __name__ == '__main__':
    olivetti_faces = fetch_olivetti_faces(data_home="F:/githubProject/AiPremiumClass/梁锐江/week4", shuffle=True)
    train_data = TensorDataset(torch.tensor(olivetti_faces.data), torch.tensor(olivetti_faces.target, dtype=torch.long))
    # print(len(train_data))
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    # model1 = SimpleModel()
    # optimizerSGD = torch.optim.SGD(model1.parameters(), lr=LR)
    # his1 = train_model(EPOCHS, model1, dataloader, loss_fn, optimizerSGD)
    # draw_train_hist(his1)
    # print("1")

    # model2 = ModelWithBN()
    # model2.train()
    # optimizerSGD2 = torch.optim.SGD(model2.parameters(), lr=LR)
    # his2 = train_model(EPOCHS, model2, dataloader, loss_fn, optimizerSGD2)
    # draw_train_hist(his2)
    # print("2")
    #
    # model3 = ModelWithDP()
    # model3.train()
    # optimizerSGD3 = torch.optim.SGD(model3.parameters(), lr=LR)
    # his3 = train_model(EPOCHS, model3, dataloader, loss_fn, optimizerSGD3)
    # draw_train_hist(his3)
    # print("3")
    #
    # model5 = ModelWithBNAndDP()
    # model5.train()
    # optimizerSGD5 = torch.optim.SGD(model5.parameters(), lr=LR)
    # his5 = train_model(EPOCHS, model5, dataloader, loss_fn, optimizerSGD5)
    # draw_train_hist(his5)
    # print("5")
    #
    # model6 = ModelWithDPAndBN()
    # model6.train()
    # optimizerSGD6 = torch.optim.SGD(model6.parameters(), lr=LR)
    # his6 = train_model(EPOCHS, model6, dataloader, loss_fn, optimizerSGD6)
    # draw_train_hist(his6)
    # print("6")

    # train_model_without_optim()
    model1 = SimpleModel()
    optimizerSGD = torch.optim.Adam(model1.parameters(), lr=LR)
    his1 = train_model(EPOCHS, model1, dataloader, loss_fn, optimizerSGD)
    draw_train_hist(his1)
    print("4")
    print("4")
