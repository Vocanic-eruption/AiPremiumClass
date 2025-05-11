import torch.nn as nn
import torch
import torch.optim as optim
import kagglehub
import os
import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split

# Download latest version
# path = kagglehub.dataset_download("utmhikari/doubanmovieshortcomments")
path = 'C:/Users/ligt/.cache/kagglehub/datasets/utmhikari/doubanmovieshortcomments/versions/7/'
print("Path to dataset files:", path)

data_path = os.path.join(path,"DMSC.csv")

data = pd.read_csv(data_path)
data = data[['Star','Comment']]

#star==1,2是正例，4，5是负例，定义转换接口
def convert_star(Star):
    if Star in [1,2]:
        return 1
    elif Star in [4,5]:
        return 0
    else:
        return -1
data['Star'] = data['Star'].apply(convert_star)

# 删除了转换为 -1的行，包括评论一起去除掉
data = data[data['Star'] != -1]

#加载停用词
stop_set = set()
def load_stop():
    with open('C:/Users/ligt/bd_AI/w07/stop_words.txt','r',encoding='utf-8') as f:
        for line in f:
            stop_set.add(line.strip())
            
    return stop_set

stop_w = load_stop()

#构建词汇表
#isse修改：vocal.add(words) 应该改为 vocal.update(words)，因为 words 是一个列表，而 add 方法只接受单个元素
vocal = set()
for comment in data['Comment']:
    comment = jieba.lcut(comment)
    words = [word for word in comment if word not in stop_w] #去掉停用词
    vocal.update(words)
    
vocal = sorted(vocal)  #按照字典顺序排序
vocal_size = len(vocal)
print(f"词汇表大小：{len(vocal)}")

# # 创建词到索引的映射，将文本转换为索引序列 0:a 1:b 
word2idx = {word:i for i ,word in enumerate(vocal)}

# 将文本转换为索引序列
text_idx = []
for ti in data['Comment']:
    t2i = jieba.lcut(ti)
    t2i_s = [ws for ws in t2i if ws not in stop_w]
    idx_seq = [word2idx[idx] for idx in t2i_s if idx in word2idx]  #将评论先分词->去停用词->根据序列表将分词转成索引序列的id然后保存
    text_idx.append(torch.tensor(idx_seq))
    

text_lable = torch.tensor(data['Star'].values)

x_train,x_val,y_train,y_val = train_test_split(text_idx, text_lable, test_size=0.2,random_state=42)

#手动实现embedding,就是生成一个大矩阵，然后每个词对应着一行维度表示词的向量
class myEmbedding(nn.Module):
    def __init__(self,vocal_size,embedding_dim):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocal_size,embedding_dim))
    def forward(self,input_idx):
        return self.embedding_matrix[input_idx]
    
class myModel(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocal_size,num_class):
        super().__init__()
        self.embedding = myEmbedding(vocal_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim,num_class)
    
    def forward(self,input_dix):
        embed = self.embedding(input_dix)
        out,_ = self.lstm(embed)
        output = self.fc(out[:,-1,:])
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
embedding_dim = 10
hidden_size = 20
num_classes = 2
LR = 1e-3
num_epochs = 100

model = myModel(embedding_dim,hidden_size, vocal_size, num_classes)
model.to(device)

#定义损失和优化器
my_loss = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), LR)

#训练函数
def train_func(model,loss,optimizer,train_data,val_data,epochs):
    model.train()
    sum_loss,correct,total =0,0,0
    for i in range(epochs):
        
        for i in range(len(train_data)):
            inputs = train_data[i].unsqueeze(0).to(device)
            lables = torch.tensor([y_train[i]]).to(device)
            
            output = model(inputs)
            loss_train = loss(output,lables)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            sum_loss += loss_train.item()
            _,pred = torch.max(output,1)
            correct += (pred == lables).sum().item()
            total += lables.size(0)
        model.eval()
        val_loss,val_correct,total_val=0,0,0
        with torch.no_grad():
            for i in len(val_data):
                in_v = val_data[i].to(device)
                l_v = torch.tensor([y_val[i]]).to(device)
                o_v = model(in_v)
                lo_v = loss(o_v,l_v)
                
                val_loss += lo_v.item()
                _,pred_v = torch.max(o_v,1)
                val_correct += (pred_v == l_v).sum().item()
                total_val += l_v.size(0)
        val_acc = val_correct/total_val
        val_avgloss = val_loss/len(val_data)
        print(f'epochs:{i+1},ACC:{val_acc:.3f},avgloss:{val_avgloss:3f}')
        
            

train_func(model,my_loss,optim,x_train,y_train,num_epochs)

#保存权重
torch.save(model.state_dict(), 'model.pth')


#构建测试数据
test_data = ['电影太好看了，是我看过最好看的电影了！']
test_data = jieba.lcut(test_data[0])
test_data = [ws for ws in test_data if ws not in stop_w]
test_data = [word2idx[idx] for idx in test_data if idx in word2idx]
test_data = torch.tensor(test_data)
test_data = test_data.unsqueeze(0).to(device)
test_lable = torch.tensor([0]).to(device)

#预测
with torch.no_grad():
    model.eval()
    output = model(test_data)
    _,pred = torch.max(output,1)
    print(pred)


