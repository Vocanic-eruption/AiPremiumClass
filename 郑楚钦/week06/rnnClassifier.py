import torch
rnn = torch.nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))


class RNN_Classifier(torch.nn.Module):

    def __init__(self, model_type: torch.nn.RNNBase, bidirectional: bool, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.rnn = model_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            bias=True,
            batch_first=True
        )
        in_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = torch.nn.Linear(in_size, output_size)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        output = outputs[:, -1, :]
        output = self.linear(output)
        return output

    def predict(self, x):
        output = self.forward(x)
        return torch.argmax(output, dim=1)


if __name__ == '__main__':
    model = RNN_Classifier(torch.nn.LSTM, False, 28, 64, 10, 5)
    print(torch.nn.LSTM.__name__)
    print(model)
    print(model(torch.randn(1, 28, 28)).shape)
