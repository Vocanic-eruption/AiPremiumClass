import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from homework_1 import *

class ModifiedEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, merge_mode='concat'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.merge_mode = merge_mode
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        
        # 第一层LSTM
        output1, (h1, c1) = self.lstm1(embedded, hidden[0:2])
        # 第二层LSTM
        output2, (h2, c2) = self.lstm2(output1, hidden[2:4])
        
        # 合并方式
        if self.merge_mode == 'concat':
            final_h = torch.cat([h1, h2], dim=2)
            final_c = torch.cat([c1, c2], dim=2)
        elif self.merge_mode == 'add':
            final_h = h1 + h2
            final_c = c1 + c2
            
        return output2, (final_h, final_c)
    
    
# 初始化两种Encoder
encoder_concat = ModifiedEncoder(vocab_size, 256, 512, 'concat').to(device)
encoder_add = ModifiedEncoder(vocab_size, 256, 512, 'add').to(device)

# 训练时需要调整Decoder的输入维度
decoder_concat = Decoder(vocab_size, 256, 1024).to(device)  # concat模式hidden_size翻倍
decoder_add = Decoder(vocab_size, 256, 512).to(device)    
    
def train(epochs=1):
    encoder_add.train()
    decoder_add.train()
    
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

train(epochs=5)
    
    