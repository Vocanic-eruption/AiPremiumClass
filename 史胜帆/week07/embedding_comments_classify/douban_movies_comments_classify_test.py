import torch
import torch.nn as nn
import pickle
import jieba
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence # 长度不同的张量填充为相同长度 length pad_sequence要求传入数据是张量

class Comments_Classifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx = 0) # padding_idx 指定pad的索引值 默认none 此处填上的好处在于 避免填充值0索引对应的向量参与到模型的训练的前向传播和反向传播中
        self.rnn = nn.LSTM(embedding_dim,hidden_size,batch_first = True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,input_idx):
        # input_idx: (batch_size,sequence_len)
        # embedding: (batch_size,sequene_len,embedding_dim)
        embedding = self.embedding(input_idx)
        # output: (batch_sizem,seq_len,hidden_size)
        output,(hidden,_) =  self.rnn(embedding)
        output = self.fc(output[:,-1,:])
        return output

# 测试模型
comment1 = "very good!"
comment2 = "这部电影非常好看！@！"
comment3 = "没意思 看不懂"

# 加载词典
wd2idx = torch.load('comments_vocab.pth')


# 评论转索引
comment1_idx = torch.tensor([wd2idx.get(word,wd2idx['UNK']) for word in jieba.lcut(comment1)])
comment2_idx = torch.tensor([wd2idx.get(word,wd2idx['UNK']) for word in jieba.lcut(comment2)])
comment3_idx = torch.tensor([wd2idx.get(word,wd2idx['UNK']) for word in jieba.lcut(comment3)])

# 索引序列转符合模型输入形状的tensor 添加batch维度
cmt1 = comment1_idx.unsqueeze(0) 
cmt2 = comment2_idx.unsqueeze(0)
cmt3 = comment3_idx.unsqueeze(0)


# 加载模型
vocab_size = len(wd2idx)
embedding_dim = 100
hidden_size = 128
num_classes = 2
LR = 0.01

model = Comments_Classifier(len(wd2idx),embedding_dim,hidden_size,num_classes)
model.load_state_dict(torch.load('comments_classifier.pth'))

# 模型测试
with torch.no_grad():
    pred1 = model(cmt1)
    pred2 = model(cmt2)
    pred3 = model(cmt3)
    
    pred1 = torch.argmax(pred1,dim = 1).item()
    pred2 = torch.argmax(pred2,dim = 1).item()
    pred3 = torch.argmax(pred3,dim = 1).item()

    print(f'测试1 预测结果:{pred1}')
    print(f'测试2 预测结果:{pred2}')
    print(f'测试3 预测结果:{pred3}')

# 预测结果全为 0 不太好 可以原因是 评论数据集里 负向评论 negative 太多

