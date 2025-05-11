import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class RNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=64,
            hidden_size=128,
            bias=True,
            num_layers=5,
            batch_first=True
        )
        self.fc = torch.nn.Linear(128, 40)

    def forward(self, x):
        output, h_t = self.rnn(x)
        return self.fc(output[:, -1, :])


class RNNClassifierWithOutLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=64,
            hidden_size=40,
            bias=True,
            num_layers=5,
            batch_first=True
        )

    def forward(self, x):
        output, h_t = self.rnn(x)
        return output[:, -1, :]


class GRUClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=64,
            hidden_size=128,
            bias=True,
            num_layers=5,
            batch_first=True
        )
        self.fc = torch.nn.Linear(128, 40)

    def forward(self, x):
        output, h_t = self.gru(x)
        return self.fc(output[:, -1, :])


class LSTMClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=128,
            bias=True,
            num_layers=5,
            batch_first=True
        )
        self.fc = torch.nn.Linear(128, 40)

    def forward(self, x):
        output, (h_0, c_0) = self.lstm(x)
        return self.fc(output[:, -1, :])


class BiRNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=128,
            bias=True,
            num_layers=5,
            bidirectional=True,
            batch_first=True
        )
        self.fc = torch.nn.Linear(128, 40)

    def forward(self, x):
        output, (h_0, c_0) = self.lstm(x)
        output = output[..., :128] + output[..., 128:]
        return self.fc(output[:, -1, :])


def get_faces_data():
    faces = fetch_olivetti_faces(data_home="./data/fetch_olivetti_faces", shuffle=True)

    images = faces['images']
    target = faces['target']

    train_data = TensorDataset(torch.tensor(images), torch.tensor(target, dtype=torch.long))

    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    return train_dl


def train_model(model, dirName):
    writer = SummaryWriter(dirName)

    train_dl = get_faces_data()

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    global_step = 0
    for epoch in range(100):
        for image_batch, target_batch in train_dl:
            image_batch, target_batch = image_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            predict = model(image_batch)
            loss = loss_function(predict, target_batch)
            loss.backward()
            optimizer.step()
            writer.add_scalar('simple rnn loss', loss, global_step)
            global_step += 1

    writer.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RNNClassifier()
    # model = RNNClassifierWithOutLinear()
    # model = GRUClassifier()
    # model = LSTMClassifier()
    model = BiRNNClassifier()

    model.to(device)
    dirName = "./runs/biRnn"
    train_model(model, dirName)
