####################jieba中文分词##############################
#分词
import jieba
strs=['我来到北京清华大学','乒乓球拍卖完了','中国科学技术大学']
for str in strs:
    seg_list=jieba.lcut(str,use_paddle=True)
    print('Paddle Mode:'+'/'.join(list(seg_list)))

seg_list=jieba.cut('我来到北京清华大学',cut_all=True)
print("Full Mode:"+"/".join(seg_list))
seg_list=jieba.cut('我来到北京清华大学',cut_all=False)
print("Default Mode:"+"/".join(seg_list))
seg_list=jieba.cut("他来到了网易杭研大厦")
print(",".join(seg_list))
seg_list=jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print(",".join(seg_list))

#词性标注
import jieba
import jieba.posseg as pseg
words=pseg.cut('我爱北京天安门')
for word,flag in words:
    print("%s%s"%(word,flag))

#Tokenize
result=jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print("word %s\\t\\t start: %d \\t\\t end:%d" % (tk[0],tk[1],tk[2]))
result=jieba.tokenize(u'永和服装饰品有限公司',mode='search')
for tk in result:
    print("word %s\\t\\t start: %d \\t\\t end:%d" % (tk[0],tk[1],tk[2]))

##################################文本分类器sentencepiece##################
#安装
pip install sentencepiece
#模型训练
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input=files,model_prefix='words_technology',vocab_size=16000
)
#分词
import sentencepiece as spm

sp=spm.SentencePieceProcessor(model_file=model_path)
out=sp.encode(
    '所谓"低代码"或"零代码",指的是不编写或少编写代码，就能完成开发任务。这既有助于扩大用户规模，获得更大的市场，也有助于程序员减轻工作负荷，避免重复劳动。'
)
sp.id_to_piece(out)
#########################文本结构化转换#####################
class Vocabulary:
    def __init__(self,vocab):
        self.vocab=vocab
    @classmethod
    def from_documents(cls,documents):
        no_repeat_tokens=set()
        for cmt in documents:
            no_repeat_tokens_update(lis(cmt))
        tokens=['PAD','UNK']+list(no_repeat_tokens)
        vocab={ tk:i for i,tk in enumerate(tokens)}
        return cls(vocab)

class CommentDataset:
    def __init__(self,comments,labels,vocab):
        self.comments=comments
        self.labels=labels
        self.vocab=vocab
    def __getitem__(self,index):
        token_index=[]
        for tk in list(self.comments[index]):
            if tk!='':
                tk_idx=self.vocab.get(tk,0)
                token_index.append(tk_idx)
    index_tensor=torch.zeros(size=(125,),dtype=torch.int)
    for i in range(len(token_index)):
        index_tensor[i]=token_index[i]
    return index_tensor,torch.tensor(self.labels[index])
    def __len__(self):
        return len(self.labels)

from torch.nn.utils.rnn import pad_sequence

devcie=torch.device("cuda" if torch.cuda.is_available()else "cpu")

text_pipeline=lambda x:vocab(tokenizer(x))
label_pipeline=lambda x:int(x)-1

def collate_batch(batch):
    label_list,text_list=[],[]
    for (_label,_text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text=torch.tensor(text_pipeline(_text),dtype=torch.int64)
        text_list.append(processed_text)
    label_list=torch.tensor(label_list,dtype=torch.int64)
    text_list=pad_sequence(text_list,batch_first=True,padding_value=vocab['PAD'])
    return label_list.to(device),text_list.to(device)

dataloader=Dataloader(train,batch_size=8,shuffle=True,collate_fn=collate_batch)


#词向量Embedding
from torch.nn import EmbeddingBag
emb=Embedding(vocab_size,hiddent_size,padding_idx=0)

print(vocab[;'PAD'])
print(vocab['UNK'])

class SomeModel(nn.Moduel):
    def __init__(self,vocab_size,emb_hidden_size,rnn_hidden_size,num_layers,num_class):
        super(SomeModel,self).__init__()
        self.embedding=nn.Embedding(vocab_size,emb_hidden_size,padding_dix=0)
        self.rnn=nn.LSTM(
            input_size=emb_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out=nn.linear(rnn_hidden_size,num_class)
    def forward(self,x):
        out=self.embedding(x)
        r_out,(c_n,h_n)=self.rnn(out)
        out=self.out(r_out[:,-1,:])
        return out

model=SomeModel(len(vocab),128,128,1,5)
model.to(device)
for label,text in dataloader:
    text.to(device)
    out=model(text)
    print(out)
    break


####################TextRNN######################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

from comment_classifier_model import CommentsClassifier
from comment_dataset import CommentDataset,Vocabulary

class SummaryWrapper:
    def __init__(self):
        self.writer=SummaryWriter()
        self.train_loss_cnt=0
    def train_loss(self,func):
        def wrpper(*args):
            result=func(*args)
            self.writer.add+scalar('train_loss',result,self.train_loss_cnt)
            self.train_loss_cnt+=1
            return wrpper

sw=SummaryWrapper()

def train(model,train_dl,criterion,optimizer):
    model.train()
    tpbar=tqdm(train_dl)
    for tokens,labels in tpbar:
        loss=train_step(model,tokens,labels,criterion)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        tpbar.set_description(f'epoch:{epoch+1}train_loss:{loss.item():.4f}')

@sw.train_loss
def train_step(model,tokens,labels,criterion):
    tokens,labels=tokens.to(device),labels.to(device)
    logits=model(tokens)
    loss=criterion(logits,labels)
    return loss
if __name__ =='__main__':
    BATCH_SIZE=32
    EPOCHS=10
    EMBEDDING_SIZE=200
    RNN_HIDDEN_SIZE=100
    LEARIN_RATE=1e-3
    NUM_LABELS=2
    device=torch.device('cuda'if torch.cuda.is_available() else  torch.mps.is_available() else'cpu')


#数据准备
import pickle
with open('comments.bin','rb') as f:
    comments,labels=pickle.load(f)

vocab=Vocabulary.from_documents(comments)

#数据拆分
(x_train,y_train),(x_test,y_test)=train_test_split(comments,labels)

#自定义Dataset处理文本数据转换
train_ds=CommentDataset(comments,y_train,vocab.vocab)
train_dl=DataLoader(train_ds,batch_size=10,shuffle=True)

#模型构建
model=CommentsClassifier(
    vocab_size=len(train_ds.vocab),
    emb_size=EMBEDDING_SIZE
    rnn_hidden_size=RNN_HIDDEN_SIZE,
    num_labels=NUM_LABELS
)
model.t0(device)

#loss function、optimizer
optimizer=optim.Adam(model.parameters(),lr=LEARIN_RATE)
criterion=nn.CrossEntropyLoss()

#训练
for epoch in range(EPOCHS):
    train(model,train_dl,criterion,optimizer)

torch.save(
    {'model_state':model.state_dictk,
     'model_vocab':vocab})




#1. 使用豆瓣电影评论数据完成文本分类处理：文本预处理，加载、构建词典。（评论得分1～2	表示positive取值：1，评论得分4～5代表negative取值：0）
#https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
import pandas as pd
import jieba
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import os

# 下载数据并读取（请确保你已经从kaggle下载好数据并放置在合适的路径下）
data_path = 'douban_movie_comments.csv'  # 请根据实际路径修改
df = pd.read_csv(data_path)

# 数据预处理，将评论得分转换为标签（1～2表示positive取值：1，4～5代表negative取值：0）
def convert_label(score):
    if score in [1, 2]:
        return 1
    elif score in [4, 5]:
        return 0
    else:
        return None  # 对于其他情况，这里简单处理为None，你可以根据需求进一步处理

df['label'] = df['score'].apply(convert_label)
df = df.dropna(subset=['label'])  # 去除标签为None的行
comments = df['comment'].tolist()
labels = df['label'].tolist()

# 使用jieba进行分词
def tokenize_with_jieba(text):
    seg_list = jieba.lcut(text, use_paddle=True)
    return seg_list

tokenized_comments_jieba = [tokenize_with_jieba(cmt) for cmt in comments]

# sentencepiece模型训练（假设你想将数据临时存储为文本文件进行训练，训练完成后可以删除）
train_text = '\n'.join(comments)
with open('train.txt', 'w', encoding='utf-8') as f:
    f.write(train_text)

spm.SentencePieceTrainer.train(
    input='train.txt', model_prefix='words_technology', vocab_size=16000
)
os.remove('train.txt')  # 删除临时文件

# sentencepiece分词
sp = spm.SentencePieceProcessor(model_file='words_technology.model')
tokenized_comments_sp = [sp.encode(cmt) for cmt in comments]

# 构建词典（这里以jieba分词结果构建为例，你也可以使用sentencepiece的结果构建）
class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        no_repeat_tokens = set()
        for cmt in documents:
            no_repeat_tokens.update(cmt)
        tokens = ['PAD', 'UNK'] + list(no_repeat_tokens)
        vocab = {tk: i for i, tk in enumerate(tokens)}
        return cls(vocab)

vocab = Vocabulary.from_documents(tokenized_comments_jieba)

# 自定义数据集类
class CommentDataset(Dataset):
    def __init__(self, comments, labels, vocab):
        self.comments = comments
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, index):
        token_index = []
        for tk in self.comments[index]:
            tk_idx = self.vocab.vocab.get(tk, 0)
            token_index.append(tk_idx)
        index_tensor = torch.zeros(size=(125,), dtype=torch.int)
        for i in range(len(token_index)):
            index_tensor[i] = token_index[i]
        return index_tensor, torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)

# 数据加载器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_pipeline = lambda x: [vocab.vocab.get(tk, 0) for tk in x]
label_pipeline = lambda x: x

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab.vocab['PAD'])
    return label_list.to(device), text_list.to(device)

train_dataset = CommentDataset(tokenized_comments_jieba, labels, vocab)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

# 定义模型
class SomeModel(nn.Module):
    def __init__(self, vocab_size, emb_hidden_size, rnn_hidden_size, num_layers, num_class):
        super(SomeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=emb_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, x):
        out = self.embedding(x)
        r_out, (c_n, h_n) = self.rnn(out)
        out = self.out(r_out[:, -1, :])
        return out

model = SomeModel(len(vocab.vocab), 128, 128, 1, 2)  # 因为是二分类，num_class为2
model.to(device)

# 简单的训练循环示例（这里仅为了展示，实际需要更多训练轮次和优化）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(1):
    running_loss = 0.0
    for label, text in dataloader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

    #2. 加载处理后文本构建词典、定义模型、训练、评估、测试。
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np


# 假设数据已经经过处理，存储在文件中，读取数据
data_path = 'processed_data.csv'  # 请根据实际路径修改
df = pd.read_csv(data_path)
comments = df['comment'].tolist()
labels = df['label'].tolist()


# 构建词典
class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        no_repeat_tokens = set()
        for cmt in documents:
            token_list = cmt.split()  # 假设文本已经是分词后的，以空格分隔
            no_repeat_tokens.update(token_list)
        tokens = ['PAD', 'UNK'] + list(no_repeat_tokens)
        vocab = {tk: i for i, tk in enumerate(tokens)}
        return cls(vocab)


vocab = Vocabulary.from_documents(comments)


# 自定义数据集类
class CommentDataset(Dataset):
    def __init__(self, comments, labels, vocab):
        self.comments = comments
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, index):
        token_index = []
        token_list = self.comments[index].split()
        for tk in token_list:
            tk_idx = self.vocab.vocab.get(tk, 0)
            token_index.append(tk_idx)
        index_tensor = torch.zeros(size=(125,), dtype=torch.int)
        for i in range(len(token_index)):
            index_tensor[i] = token_index[i]
        return index_tensor, torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)


# 数据加载器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_pipeline = lambda x: [vocab.vocab.get(tk, 0) for tk in x.split()]
label_pipeline = lambda x: x


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab.vocab['PAD'])
    return label_list.to(device), text_list.to(device)


train_dataset = CommentDataset(comments, labels, vocab)
dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)


# 定义模型
class SomeModel(nn.Module):
    def __init__(self, vocab_size, emb_hidden_size, rnn_hidden_size, num_layers, num_class):
        super(SomeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=emb_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, x):
        out = self.embedding(x)
        r_out, (c_n, h_n) = self.rnn(out)
        out = self.out(r_out[:, -1, :])
        return out


model = SomeModel(len(vocab.vocab), 128, 128, 1, 2)
model.to(device)


# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for label, text in dataloader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')


# 评估模型（假设验证集数据格式与训练集相同）
val_data_path = 'val_data.csv'  # 请根据实际路径修改
val_df = pd.read_csv(val_data_path)
val_comments = val_df['comment'].tolist()
val_labels = val_df['label'].tolist()

val_dataset = CommentDataset(val_comments, val_labels, vocab)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_batch)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for label, text in val_dataloader:
        output = model(text)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Validation Accuracy: {100 * correct / total}%')


# 测试模型（假设测试集数据格式与训练集相同）
test_data_path = 'test_data.csv'  # 请根据实际路径修改
test_df = pd.read_csv(test_data_path)
test_comments = test_df['comment'].tolist()
test_labels = test_df['label'].tolist()

test_dataset = CommentDataset(test_comments, test_labels, vocab)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_batch)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for label, text in test_dataloader:
        output = model(text)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')



#3. 尝试不同分词工具进行文本分词，观察模型训练结果
import pandas as pd
import jieba
import thulac  # 清华分词工具，需先安装 pip install thulac
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np


# 读取数据
data_path = 'raw_data.csv'  # 请根据实际路径修改
df = pd.read_csv(data_path)
comments = df['comment'].tolist()
labels = df['label'].tolist()


# 使用jieba分词
def tokenize_with_jieba(text):
    seg_list = jieba.lcut(text)
    return " ".join(seg_list)


jieba_tokenized_comments = [tokenize_with_jieba(cmt) for cmt in comments]


# 使用清华分词工具thulac分词
thu = thulac.thulac()
def tokenize_with_thulac(text):
    result = thu.cut(text, text=True)
    return result


thulac_tokenized_comments = [tokenize_with_thulac(cmt) for cmt in comments]


# 分别构建词典（以jieba分词结果构建为例，thulac类似）
class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        no_repeat_tokens = set()
        for cmt in documents:
            token_list = cmt.split()
            no_repeat_tokens.update(token_list)
        tokens = ['PAD', 'UNK'] + list(no_repeat_tokens)
        vocab = {tk: i for i, tk in enumerate(tokens)}
        return cls(vocab)


jieba_vocab = Vocabulary.from_documents(jieba_tokenized_comments)
thulac_vocab = Vocabulary.from_documents(thulac_tokenized_comments)


# 定义数据集类（以jieba分词结果构建的数据集为例，thulac类似）
class CommentDataset(Dataset):
    def __init__(self, comments, labels, vocab):
        self.comments = comments
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, index):
        token_index = []
        token_list = self.comments[index].split()
        for tk in token_list:
            tk_idx = self.vocab.vocab.get(tk, 0)
            token_index.append(tk_idx)
        index_tensor = torch.zeros(size=(125,), dtype=torch.int)
        for i in range(len(token_index)):
            index_tensor[i] = token_index[i]
        return index_tensor, torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)


# 数据加载器（以jieba分词结果构建的数据集为例，thulac类似）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
jieba_text_pipeline = lambda x: [jieba_vocab.vocab.get(tk, 0) for tk in x.split()]
thulac_text_pipeline = lambda x: [thulac_vocab.vocab.get(tk, 0) for tk in x.split()]
label_pipeline = lambda x: x


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab.vocab['PAD'])
    return label_list.to(device), text_list.to(device)


jieba_train_dataset = CommentDataset(jieba_tokenized_comments, labels, jieba_vocab)
jieba_dataloader = DataLoader(jieba_train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

thulac_train_dataset = CommentDataset(thulac_tokenized_comments, labels, thulac_vocab)
thulac_dataloader = DataLoader(thulac_train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)


# 定义模型
class SomeModel(nn.Module):
    def __init__(self, vocab_size, emb_hidden_size, rnn_hidden_size, num_layers, num_class):
        super(SomeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_hidden_size, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=emb_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, x):
        out = self.embedding(x)
        r_out, (c_n, h_n) = self.rnn(out)
        out = self.out(r_out[:, -1, :])
        return out


# 使用jieba分词结果训练模型
jieba_model = SomeModel(len(jieba_vocab.vocab), 128, 128, 1, 2)
jieba_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(jieba_model.parameters())

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for label, text in jieba_dataloader:
        optimizer.zero_grad()
        output = jieba_model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Jieba Epoch {epoch + 1}, Loss: {running_loss / len(jieba_dataloader)}')


# 使用thulac分词结果训练模型
thulac_model = SomeModel(len(thulac_vocab.vocab), 128, 128, 1, 2)
thulac_model.to(device)

optimizer = torch.optim.Adam(thulac_model.parameters())

for epoch in range(num_epochs):
    running_loss = 0.0
    for label, text in thulac_dataloader:
        optimizer.zero_grad()
        output = thulac_model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Thulac Epoch {epoch + 1}, Loss: {running_loss / len(thulac_dataloader)}')


# 可以进一步进行评估和测试，观察不同分词工具下模型的性能差异
