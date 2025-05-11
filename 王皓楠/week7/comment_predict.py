import csv
import torch
import numpy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba 
import pickle
import os 
import sys 
import io
from comments_deal import CommentClassifier
#此文件负责加载模型，进行评估测试

embedding_dim = 100
hidden_size = 128
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载词典
vocab=torch.load('comments_vocab.pth')
# 测试模型
comment1 = '我喜欢看这部电影'
comment2 = '垃圾电影我完全不想看'
comment3='这部电影五星好评！非常喜欢！他的艺术价值非常高很伟大强烈推荐'

# 将评论转换为索引
comment1_idx=torch.tensor([vocab.get(word,vocab['UNK'])for word in jieba.lcut(comment1)])
comment2_idx=torch.tensor([vocab.get(word,vocab['UNK'])for word in jieba.lcut(comment2)])                    
comment3_idx=torch.tensor([vocab.get(word,vocab['UNK'])for word in jieba.lcut(comment3)])

# 将评论增加维度，放入模型进入预测,添加batch维度
comment1_idx=comment1_idx.unsqueeze(0).to(device)
comment2_idx=comment2_idx.unsqueeze(0).to(device)
comment3_idx=comment3_idx.unsqueeze(0).to(device)

# 加载模型
model=CommentClassifier(len(vocab),embedding_dim,hidden_size,num_classes)
model.load_state_dict(torch.load('comments_classifier.pth'))
model.to(device)

# 模型推理
pred1=model(comment1_idx)
pred2=model(comment2_idx)
pred3=model(comment3_idx)

# 取最大值的索引作为预测结果
pred1 = torch.argmax(pred1, dim=1).item()
pred2 = torch.argmax(pred2, dim=1).item()
pred3 = torch.argmax(pred3, dim=1).item()
print(f'评论1预测结果: {pred1}')
print(f'评论2预测结果: {pred2}')
print(f'评论3预测结果: {pred3}')