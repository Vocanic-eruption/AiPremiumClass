from fetch_olivetti_faces_rnn import get_faces_data
import torch
import torch.nn as nn

class GruRnnClassifier(nn.modules):
    def __init__(self):
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            bias=True,
            num_layers=5,
            batch_first=True
        )
        self.fc = nn.Linear(128, 40)

    def forward(self, x):
        pass


if __name__ == '__main__':
    train_dl = get_faces_data()
