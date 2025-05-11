import torch
import torch.nn as nn

class TorchNNBN(nn.Module):
    def __init__(self):
        super(TorchNNBN, self).__init__()
        self.linear1 = nn.Linear(64 * 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 40)
        self.drop = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear3(out)
        return out


if __name__ == '__main__':
    model = TorchNNBN()
    input_data = torch.randn((40,64*64))
    predict = model(input_data)
    print(predict.shape)

