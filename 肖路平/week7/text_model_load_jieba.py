import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence #长度不同的张量进行填充
import jieba
from text_classifier_jieba import Comment_classifier, build_from_doc


vocab = torch.load('E:\\workspacepython\\AiPremiumClass\\vocab.pth')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#测试模型
comment1 = "这部电影真好看！全程无尿点"
comment2 = "这个电影太差了！很无聊，演员演技也很差"

comment1_idx= torch.tensor([vocab.get(word,vocab['UNK']) for word in jieba.lcut(comment1)])
comment2_idx= torch.tensor([vocab.get(word,vocab['UNK']) for word in jieba.lcut(comment2)])

comment1_idx = comment1_idx.unsqueeze(0).to(device)
comment2_idx = comment2_idx.unsqueeze(0).to(device)

#加载模型
vocab_size=len(vocab)
embedding_dim=100
hidden_size=128
num_classes=2
model=Comment_classifier(vocab_size,embedding_dim,hidden_size,num_classes)
model.load_state_dict(torch.load('E:\\workspacepython\\AiPremiumClass\\commetn_classifier.pth'))
model.to(device) 
pred1 = model(comment1_idx)
pred2 = model(comment2_idx)

pred1= torch.argmax(pred1,dim=1).item()
pred2= torch.argmax(pred2,dim=1).item()

print("评论1：",pred1,"\n评论2：",pred2)