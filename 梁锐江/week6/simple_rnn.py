import torch

rnn = torch.nn.RNN(
    input_size=28,
    hidden_size=50,
    num_layers=1,
    bias=True
)

x = torch.randn(5, 28, 28)

if __name__ == '__main__':
    out, h_n = rnn(x)
    print(out.shape)
    print(h_n.shape)
