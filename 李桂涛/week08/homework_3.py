import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from homework_1 import encoder,decoder,word2idx,idx2word

device = torch.device('cuda:0' if torch.is_available() else 'cpu')
embed_dim = 256
hidden_size = 256

def generate_couplet(sentence, max_len=50):
    encoder.eval()
    decoder.eval()
    
    # 处理输入
    input_seq = ['<start>'] + list(sentence) + ['<end>']
    input_idx = [word2idx.get(c, word2idx['<unk>']) for c in input_seq]
    input_tensor = torch.tensor(input_idx, device=device).unsqueeze(0)
    
    # Encoder
    with torch.no_grad():
        h0 = torch.zeros(1, 1, hidden_size).to(device)
        c0 = torch.zeros(1, 1, hidden_size).to(device)
        encoder_outputs, (hidden, cell) = encoder(input_tensor, (h0, c0))
    
    # Decoder初始化
    decoder_input = torch.tensor([[word2idx['<start>']]], device=device)
    result = []
    
    for _ in range(max_len):
        with torch.no_grad():
            output, (hidden, cell) = decoder(
                decoder_input,
                (hidden, cell),
                encoder_outputs
            )
            
        pred_id = output.argmax(1)
        result.append(pred_id.item())
        
        if pred_id == word2idx['<end>']:
            break
            
        decoder_input = pred_id.unsqueeze(1)
    
    # 转换结果
    generated = [idx2word[idx] for idx in result if idx not in {word2idx['<start>'], word2idx['<end>']}]
    return ''.join(generated)

# 测试

"""
1、晚 风 摇 树 树 还 挺 
2、愿 景 天 成 无 墨 迹 
3、丹 枫 江 冷 人 初 去 

1、晨 露 润 花 花 更 红 
2、万 方 乐 奏 有 于 阗 
3、绿 柳 堤 新 燕 复 来 
"""
print(generate_couplet("春风送暖")) 