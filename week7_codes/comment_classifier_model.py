import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CommentsClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, rnn_hidden_size, num_labels):
        super().__init__()
        # embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 字典大小 
            embedding_dim=emb_size,     # 词向量大小
            padding_idx=0               # 填充值索引 (不会参与模型训练推理)
            )     
        
        # rnn 分析token序列 unigram
        self.rnn = nn.LSTM(
            input_size=emb_size,  # 词向量大小 
            hidden_size=rnn_hidden_size,  # 参数大小
            batch_first=True)
        
        # 分类预测
        self.classifer = nn.Linear(rnn_hidden_size, num_labels)

    def forward(self, input_data):
        # shape [batch,token_len] -> [batch, token_len, emb_size]
        out = self.embedding(input_data)
        # shape [batch,token_len,emb_size] -> [batch, token_len, rnn_hidden_size]
        output, _ = self.rnn(out)
        # shape [batch, rnn_hidden_size] -> [batch, num_class]
        return self.classifer(output[:,-1,:])
    

if __name__ == '__main__':
    
    input_data = torch.randint(1,10, size=(10,12))

    model = CommentsClassifier(vocab_size=10, emb_size=20, rnn_hidden_size=20, num_labels=2)

    logits = model(input_data)

    print(logits.shape)