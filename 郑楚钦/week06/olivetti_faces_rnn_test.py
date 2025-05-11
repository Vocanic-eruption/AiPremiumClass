from torch.utils.tensorboard import SummaryWriter
from rnnClassifier import RNN_Classifier
from sklearn.datasets import fetch_olivetti_faces
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def split_data(data, test_size):
    # split 数据集
    train_size = len(data) - test_size
    return torch.utils.data.random_split(data, [train_size, test_size])


def to_tensor(data, device=device):
    # 转化为张量
    return torch.tensor(data).to(device=device)


def to_dataloader(data, batch_size):
    # 转化为DataLoader
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)


def to_dataloaders(data, target, batch_size, test_size):
    # 分割数据集并转为DataLoader
    dataset = [(features, label)
               for features, label in zip(to_tensor(data), to_tensor(target))]
    print(len(dataset))
    train_dataset, test_dataset = split_data(dataset, test_size)
    return to_dataloader(train_dataset, batch_size), to_dataloader(test_dataset, batch_size)


def get_dataloaders(batch_size=30, test_size=100):
    olivetti_faces = fetch_olivetti_faces(data_home='./olivetti', shuffle=True)
    print(olivetti_faces.data.shape, olivetti_faces.target.shape,
          olivetti_faces.images.shape)
    return to_dataloaders(olivetti_faces.images, olivetti_faces.target, batch_size, test_size)


def trainAndTest(model_type: torch.nn.RNNBase, bidirectional: bool = False, epoches=100, lr=0.001, hidden_size: int = 128,  num_layers: int = 2):

    train_dl, test_dl = get_dataloaders()
    model_name = ('bi_' if bidirectional else '') + model_type.__name__
    model = RNN_Classifier(model_type, bidirectional, 64, hidden_size,
                           40, num_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        model.train()
        for batch_idx, (features, target) in enumerate(train_dl):
            optimizer.zero_grad()
            ouput = model(features)
            loss = loss_fn(ouput, target.to(dtype=torch.long))
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.add_scalars('loss', {model_name: loss.item()},
                               epoch * len(train_dl) + batch_idx)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, target in test_dl:
                output = model.predict(features.squeeze())
                total += target.size(0)
                correct += (output == target).sum().item()
            accuracy = 100 * correct / total
            writer.add_scalars('accuracy', {model_name: accuracy}, epoch)
    torch.save(model, f'model_{model_name}.pth')
    torch.save(model.state_dict(), f'model_{model_name}_params.pth')


if __name__ == '__main__':
    writer = SummaryWriter("logs")
    trainAndTest(torch.nn.RNN, bidirectional=True)
    trainAndTest(torch.nn.RNN)
    trainAndTest(torch.nn.GRU, bidirectional=True)
    trainAndTest(torch.nn.GRU)
    trainAndTest(torch.nn.LSTM, bidirectional=True)
    trainAndTest(torch.nn.LSTM)
    writer.close()
