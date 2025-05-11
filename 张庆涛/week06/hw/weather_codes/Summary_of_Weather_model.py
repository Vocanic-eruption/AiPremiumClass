import torch.nn as nn
class SummaryOfWeater(nn.Module):
  def __init__(self, input_size, hidden_size, output_size,num_layers=1):
    super().__init__()
    self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True, dropout=0.2)
    self.fc = nn.Linear(hidden_size,output_size)

  def forward(self, x):
    if x.dim() == 2:
      x = x.unsqueeze(0)
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out