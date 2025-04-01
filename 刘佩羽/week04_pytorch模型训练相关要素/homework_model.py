import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        # 初始化函数
        super().__init__()
        # 调用父类的初始化函数
        self.linear1 = nn.Linear(in_features=4096, out_features=2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.linear2 = nn.Linear(in_features=2048, out_features=1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.linear3 = nn.Linear(in_features=1024, out_features=40)
        # self.bn3 = nn.BatchNorm1d(512)

        # self.linear4 = nn.Linear(in_features=512, out_features=256)
        # self.bn4 = nn.BatchNorm1d(256)

        # self.linear5 = nn.Linear(in_features=256, out_features=128)
        # self.bn5 = nn.BatchNorm1d(128)

        # self.linear6 = nn.Linear(in_features=128, out_features=40)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # out = self.linear3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        # out = self.linear4(out)
        # out = self.bn4(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        # out = self.linear5(out)
        # out = self.bn5(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        out = self.linear3(out)
        final = out
        return final


if __name__ == '__main__':
    model = NN()
    print(type(model))
    print((model))
