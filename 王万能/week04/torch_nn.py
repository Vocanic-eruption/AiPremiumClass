import torch
import torch.nn as nn

class TorchNN(nn.Module):
    def __init__(self):
        super(TorchNN, self).__init__()
        self.linear1 = nn.Linear(64 * 64, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 40)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        return out


if __name__ == '__main__':
    model = TorchNN()
    input_data = torch.randn((40,64*64))
    predict = model(input_data)
    print(predict.shape)

