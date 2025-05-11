import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class CommentsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim, 
            padding_idx=0)  # padding_idx=0
        
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size, 
            batch_first=True)
        self.classifer = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, _ = self.rnn(embedded)
        
        return self.classifer(output[:, -1, :])  # 取最后一个时间步的输出
    
if __name__ == '__main__':
    # 加载训练数据
    input_ids = torch.randint(0, 1000, (16, 128))
    
    model = CommentsClassifier(
        vocab_size=1000,
        embedding_dim=128,
        hidden_size=64,
        num_labels=2
    )
    
    logits = model(input_ids)
    
    print(logits.shape)