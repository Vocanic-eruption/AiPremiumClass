import torch

rnn = torch.nn.RNN(
    input_size=28,
    hidden_size=50, #隐藏层神经元数量w_ht[50,28],w_hh[50,50]
    bias=True,
    batch_first=True,
)
print(rnn)
# shape[batch,times,features]
X = torch.randn(10, 28, 28) #10样本 28t 28的x
outputs,last_outputs = rnn(X)
print(outputs.shape)
print(last_outputs.shape)