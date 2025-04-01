import torch.nn as nn
import torch
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, 10)
        self.bn3 = nn.BatchNorm1d(10)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear3(out)
        out = self.bn3(out)
        final = self.activation(out)
        return torch.softmax(final, dim=1)






if __name__ == '__main__':
    model = TorchModel()
    input = torch.randn(12, 784)
    final = model(input)
    print(final.shape)