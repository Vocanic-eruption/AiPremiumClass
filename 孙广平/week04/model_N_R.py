import torch
import torch.nn as nn
from zmq import device

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
       
        self.linear1 = nn.Linear(4096, 8192)
        self.bn1 = nn.BatchNorm1d(8192)

        self.linear2 = nn.Linear(8192, 16384)
        self.bn2 = nn.BatchNorm1d(16384)
        self.droupout = nn.Dropout()

        self.linear3 = nn.Linear(16384, 4096)
        self.bn3 = nn.BatchNorm1d(4096)

        self.linear4 = nn.Linear(4096, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.droupout = nn.Dropout()

        self.linear5 = nn.Linear(1024, 40)

        self.act = nn.ReLU() 

    def forward(self, input_tensor):
        x = input_tensor
        x = self.act(self.linear1(x))
        x = self.bn1(x)
        
        x = self.act(self.linear2(x))
        x = self.bn2(x)
        x = self.droupout(x)

        x = self.act(self.linear3(x))
        x = self.bn3(x)

        x = self.act(self.linear4(x))
        x = self.bn4(x)
        x = self.droupout(x)

        x = self.linear5(x)

        return x
    


if __name__ == '__main__':

    model = Model()
    print(model)
