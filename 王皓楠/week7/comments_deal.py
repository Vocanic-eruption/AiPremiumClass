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
# 把标准输出的编码设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir("week7")
def build_from_doc(doc):
    max_len=0
    vocab = set()
    for line in doc:
        words=jieba.lcut(line[0])
        #此处改成实际应该使用的分词方法
        for word in words:
            vocab.update(word)

    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

class CommentClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_classes):
        super().__init__()
        #构建词向量
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.rnn=nn.LSTM(embedding_dim,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)
    def forward(self,X):
        #将输入转化为词向量
        embedded=self.embedding(X)
        outputs,(hiddens,_)=self.rnn(embedded)
        output=self.fc(outputs[:,-1,:])
        return output
#这是利用DataLoader的一个机制自动补全

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #改为douban_comment.pkl
    with open('douban_comment.pkl','rb') as f:
        comments_data = pickle.load(f)
    vocab=build_from_doc(comments_data)
    emb=nn.Embedding(len(vocab),100)
    #进行填充
    def convert_data(comments_data):
        comments=[]
        votes=[]
        for comment,vote in comments_data:
            #需要改成分词方法
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment)]))
            votes.append(vote)
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])  # 填充为相同长度
        labels = torch.tensor(votes)
        # 返回评论和标签
        return commt, labels
    dataloader = DataLoader(comments_data, batch_size=4, shuffle=True, 
                            collate_fn=convert_data)  
    
    # 构建模型
    # vocab_size: 词汇表大小
    # embedding_dim: 词嵌入维度
    # hidden_size: LSTM隐藏层大小
    # num_classes: 分类数量
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    model = CommentClassifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    num_epochs=5
    for epoch in range(num_epochs):
        for i,(X,y) in enumerate(dataloader):
            X=X.to(device)
            y=y.to(device)
            output=model(X)
            loss=criterion(output,y)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    #模型保存
    torch.save(model.state_dict(), 'comments_classifier.pth')
    # 模型词典
    torch.save(vocab, 'comments_vocab.pth')




        