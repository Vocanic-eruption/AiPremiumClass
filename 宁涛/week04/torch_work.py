import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader
import torch.utils.data as Data


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.b1 = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16394, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        return x


def train_data_process():
    olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
    train_data, test_data = Data.random_split(olivetti_faces,
                                              [round(0.8 * len(olivetti_faces)), round(0.2 * len(olivetti_faces))])
    train_dataloader = DataLoader(dataset=train_data, batch_size=16, num_workers=0)
    test_dataloader = DataLoader(dataset=test_data, batch_size=16, num_workers=0)
    return train_dataloader, test_dataloader


def train_model(model, train_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_all = []
    for i in range(10):
        for img, lbl in train_dataloader:
            img = img.to(device)
            lbl = lbl.to(device)
            out_put = model(img)
            pre_lab = torch.argmax(out_put, dim=1)
            loss = loss_fn(out_put, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.append(loss.item())
            print(f'loss:{loss.item():.4f}')

    corrects = 0.0
    test_num = 0
    with torch.no_grad():
        for x, y in test_dataloader:

            x = x.to(device)

            y = y.to(device)
            model.eval()
            output = model(x)
            pre_lab = torch.argmax(output, dim=1)
            corrects += torch.sum(pre_lab == y.data)
            test_num += x.size(0)

        print("准确率：{}".format(corrects.double().item()/test_num))

if __name__ == '__main__':
    model = Model()
    train_dataloader, test_dataloader = train_data_process()
    train_model(model, train_dataloader, test_dataloader)
