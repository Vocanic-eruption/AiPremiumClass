# 接 comments_process.py 的电影评论数据处理 保存的ds_commernts数据 对应的comments.pkl
# 继续下一步的电影评论分类任务  二分类（好/坏）

import torch
import torch.nn as nn
import pickle
import jieba
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence # 长度不同的张量填充为相同长度 length pad_sequence要求传入数据是张量

# 1 加载训练数据 （语料和标签）
with open('./comments.pkl','rb') as f:
    comment_data = pickle.load(f)



# 2 构建词汇表 评论词汇的词汇表构建和未知词汇的补全 和 评论词汇长度（大型词向量矩阵）的统一
# 2.1 词汇表中所有词汇一定是不重复的 用 set
vocab = set() # add() 添加单个元素  update() 一次添加多个元素
for comment,vote in comment_data: 
    vocab.update(comment)

vocab = ['PAD','UNK'] + list(vocab) # list + 等价于  extend()
print(len(vocab))

# 可以把词汇表生成封装起来 

# class Vocabulary():
#     def __init__(self,word):
#         self.vocab = vocab
    
#     # 采用装饰器模式 @classmethod 作用在于 可以在不实例化对象的前提下直接调用对象里的函数
#     def build_from_doc(cls,doc):
#         vocab = set()
#         for comment,vote in doc:
#             vocab.update(comment)
        
#         # PAD: padding 填充 处理文档多个句子词汇个数不一致导致的无法批次训练的 问题  UNK: unknow 处理OOV问题
#         #填充值不应该参与到模型的训练中去 但是 为了批次训练 只有填充为看齐 所以取0、
#         # 一般padding到 整个文档中 最长句子词汇个数的长度 
#         vocab = ['PAD','UNK'] + list(vocab) # list() 为了有序化 set的底层是hash无序
        
#         return cls(vocab)


# vocab = Vocabulary.build_from_doc(comment_data)

# 2.2 将词汇表转化为索引
wd2idx = {word : index for index,word in enumerate(vocab)}
# print(list(wd2idx.items())[:5])

# 2.3 将索引转换为向量
# 首先 要有一个大型的所有向量的集合 Embedding(词嵌入)
# 如何理解嵌入？ 用大维度空间上的一个点 代表一个数据词汇  且 有计算的功能
emb = nn.Embedding(len(vocab),100)  # Embedding(vocab_length,embedding_dim)

# 3 转文本为词的索引序列

# 出现OOV问题 无法在词表中找到要训练的文本中的词
# 比如
# text_idx = [wd2idx[word] for word in '测试 文本 转换 为 索引 序列 😀 🥧'.split()]
# 如何处理OOV


texts_idx = []
for cmt in comment_data:
    text_idx = [wd2idx.get(word,wd2idx['UNK']) for word in cmt[0]]
    texts_idx.append(torch.tensor(text_idx)) # 索引序列 转 词向量 索引序列必须是tensor类型

#print(texts_idx[:2])

# 通过dataset 构建 dataloader
# dataloader = DataLoader(comment_data,batch_size = 32,shuffle = True)
# 如何用dataloader解决数据长度填充的问题 collate_fn方法 一种回调的方法 原理是在数据打包之后 传给模型之前 collate_fn再次加工整理
# 1 自定义数据转换方法(call_back function)回调函数 不由自己调用 由系统调用
# 该方法会在每个batch数据加载时调用
def convert_data(batch_data):
    # print("custom method invoked")
    # print(batch_data)
    #分别提取评论和标签
    comments,votes = [],[]
    for comment,vote in batch_data:
        comments.append(torch.tensor([wd2idx.get(word,wd2idx['UNK']) for word in comment]))
        votes.append(vote)   # 写成 votes.append(torch.tensor(vote)) 最后的labels是[tensor(0), tensor(0), tensor(0), tensor(0)]形式的列表 而且训练需要张量

    # 填充张量长度 padding_value 默认是0
    cmmts = pad_sequence(comments,batch_first = True,padding_value = wd2idx['PAD']) 
    labels = torch.tensor(votes)

    return cmmts,labels

dataloader = DataLoader(comment_data,batch_size = 4,shuffle = True,collate_fn = convert_data)



# # 试验一次 dataloader collate_fn
# for cmt,label in dataloader:
#     print(cmt,label)
#     break

# # # 文本索引序列转向量矩阵
# # sentence_emb = emb(texts_idx[0])
# # print(sentence_emb.shape)


# 模型的构建 包括 embeding 在模型的搭建中 此时我们的模型搭建不再是之前的单纯的网络层搭建
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

# 模型的生成
vocab_size = len(vocab)
embedding_dim = 100
hidden_size = 128
num_classes = 2

model = Comments_Classifier(len(vocab),embedding_dim,hidden_size,num_classes)
print(model)

# 模型参数
EPOCH = 5
LR = 0.01

# 损失函数和优化器
crition = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = LR)

# 开始训练
for epoch in range(EPOCH):
    for i,(cmt,label) in enumerate(dataloader):
        #前向传播
        output = model(cmt)
        # 计算损失
        loss = crition(output,label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度下降 && 参数更新
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'epoch:{epoch + 1}/{EPOCH},step:{(i + 1)}/{len(dataloader)},loss:{loss.item()}')


# 保存模型
torch.save(model.state_dict(),'comments_classifier.pth')

# 保存词典
torch.save(wd2idx,'comments_vocab.pth') # 训练完一个模型要保存对应的词典 因为不同的训练 出来的词典是不一样的
