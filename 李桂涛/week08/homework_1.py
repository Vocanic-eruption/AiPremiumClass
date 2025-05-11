import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

dir_path = 'C:/Users/ligt/.cache/kagglehub/datasets/jiaminggogogo/chinese-couplets/versions/2/'

train_data_path = 'couplet/train/in.txt'
train_lable_path = 'couplet/train/out.txt'

test_data_path = 'couplet/test/in.txt'
test_lable_path = 'couplet/test/out.txt'

vacabs_data_path = 'couplet/vocabs'

train_dp=os.path.join(dir_path,train_data_path)
train_lp=os.path.join(dir_path,train_lable_path)
test_dp=os.path.join(dir_path,test_data_path)
test_lp=os.path.join(dir_path,test_lable_path)
vacabs_path = os.path.join(dir_path,vacabs_data_path)

# 加载词汇表
with open(vacabs_path, 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f]
    
required_tokens = ['<pad>', '<start>', '<end>', '<unk>']
for token in required_tokens:
    if token not in vocab:
        vocab.insert(0, token)  # 自动添加缺失的标记
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)
# word2idx['<unk>'] = int(len(vocab)+1)
# word2idx['<pad>'] = int(len(vocab)+2)

# 自定义数据集
class CoupletDataset(Dataset):
    def __init__(self, in_path, out_path):
        self.inputs, self.targets = self.load_data(in_path, out_path)
        
    def load_data(self, in_path, out_path):
        inputs, targets = [], []
        with open(in_path, 'r', encoding='utf-8') as f_in, open(out_path, 'r', encoding='utf-8') as f_out:
            for in_line, out_line in zip(f_in, f_out):
                in_seq = ['<start>'] + list(in_line.strip()) + ['<end>']
                out_seq = ['<start>'] + list(out_line.strip()) + ['<end>']
                inputs.append(torch.LongTensor([word2idx.get(c, word2idx['<unk>']) for c in in_seq]))
                targets.append(torch.LongTensor([word2idx.get(c, word2idx['<unk>']) for c in out_seq]))
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 创建DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=word2idx['<pad>'])
    targets_pad = pad_sequence(targets, batch_first=True, padding_value=word2idx['<pad>'])
    return inputs_pad, targets_pad

train_dataset = CoupletDataset(train_dp, train_lp)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        
        hidden = decoder_hidden.permute(1, 0, 2)  # (batch, 1, hidden)
        energy = torch.tanh(self.W(encoder_outputs + hidden))
        attention = F.softmax(self.V(energy), dim=1)
        context = torch.sum(attention * encoder_outputs, dim=1)
        return context, attention

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_dim + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        
        # Attention计算
        context, _ = self.attention(hidden[0], encoder_outputs)
        context = context.unsqueeze(1)
        
        # 拼接输入
        lstm_input = torch.cat([embedded, context], dim=2)
        
        # LSTM处理
        output, hidden = self.lstm(lstm_input, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden
    
# 超参数
embed_dim = 256
hidden_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
encoder = Encoder(vocab_size, embed_dim, hidden_size).to(device)
decoder = Decoder(vocab_size, embed_dim, hidden_size).to(device)

# 优化器与损失函数
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])


writer = SummaryWriter()

def train(epochs=10):
    encoder.train()
    decoder.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 初始化Encoder隐藏状态
            h0 = torch.zeros(1, inputs.size(0), hidden_size).to(device)
            c0 = torch.zeros(1, inputs.size(0), hidden_size).to(device)
            encoder_outputs, (hidden, cell) = encoder(inputs, (h0, c0))
            
            # 初始化Decoder输入
            decoder_input = torch.full((inputs.size(0), 1), 
                                    word2idx['<start>'], 
                                    device=device)  # (batch, 1)
            
            loss = 0
            for t in range(1, targets.size(1)):
                decoder_output, (hidden, cell) = decoder(
                    decoder_input,
                    (hidden, cell),
                    encoder_outputs
                )
                
                # 计算损失
                loss += criterion(decoder_output, targets[:, t])
                
                # Teacher forcing
                decoder_input = targets[:, t].unsqueeze(1)  # (batch, 1)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() / targets.size(1)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item()/targets.size(1):.4f}')
        
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Training Loss', avg_loss, epoch)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

train(epochs=10)