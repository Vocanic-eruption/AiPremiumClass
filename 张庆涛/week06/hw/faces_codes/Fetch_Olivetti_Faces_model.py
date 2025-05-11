
import torch.nn as nn
from torch.utils.data import Dataset
# 自定义数据集类
class OlivettiDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
      
class FetchOlivettiFaces(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes, num_layers=1,module='LSTM'):
    super().__init__()
    # 根据传入的参数选择合适的 RNN 模型
    if module == 'RNN':
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    elif module == 'GRU':
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    elif module == 'LSTM':
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    elif module == 'BiRNN':
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        hidden_size *= 2
    else:
        raise ValueError(f"Unsupported RNN module: {module}")
    self.fc = nn.Linear(hidden_size, num_classes)
    
  def forward(self, x):
    if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加 sequence_length 维度
    out, h_n = self.rnn(x)
    return self.fc(out[:, -1, :])
  
if __name__ == '__main__':
    model = FetchOlivettiFaces(128, 64, 2, 1, 'LSTM')
    print(model)