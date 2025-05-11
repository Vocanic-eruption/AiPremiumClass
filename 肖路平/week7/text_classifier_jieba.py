import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.nn.utils.rnn import pad_sequence #长度不同的张量进行填充
import jieba

def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab = ['PAD','UNK'] +list(vocab) #['PAD','UNK']在开始加或者最后加都可以
    w2idx = {word:idx for idx,word in enumerate(vocab)}
    return w2idx

class Comment_classifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)#填充的索引值是0，避免参与到训练当中
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
 
    def forward(self, input_ids):
        # print(input_ids.shape) # torch.Size([32, 100])
        # print(input_ids)
        embedded = self.embedding(input_ids) # [batch_size, seq_len, embedding_dim]
        # print(embedding.shape) # torch.Size([32, 100, 100])
        # print(embedding)
        output,(hidden,_) = self.rnn(embedded) # []
        output = self.fc(output[:,-1,:])
        return output
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #加载训练语料
    with open('E:\\workspacepython\\AiPremiumClass\\doubanmovie.pkl','rb') as f:
        comments_data = pickle.load(f)

    #构建词汇表
    vocab = build_from_doc(comments_data)
    print("词汇表大小",len(vocab))
    #将索引转换成词汇表
    # 每个索引-》向量
    # 所有向量集合 embedding(词嵌入)    
    emb =nn.Embedding(len(vocab),100)#词汇表大小，向量维度

    def convert_data(batch_data):
        comments,votes=[],[]
        # 分别提取评论和标签
        for comment,vote in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            votes.append(vote)

        commt = pad_sequence(comments,batch_first=True,padding_value=vocab['PAD']) #填充0,padding_value的默认值也是0
        lables = torch.tensor(votes)
        return commt,lables
    #构建Dataset,collate_fn相当于回调方法，就是把数据传给模型之前使用自定义的方法将数据再次整理和加工
    dataloader = DataLoader(comments_data,batch_size=4,
                            shuffle=True,collate_fn=convert_data)
    #模型参数
    # vocab_size 词汇表大小
    # embedding_dim 词嵌入维度
    # hidden_size lstm 隐藏层维度
    # num_classes 分类数量
    vocab_size=len(vocab)
    embedding_dim=100
    hidden_size=128
    num_classes=2

    model = Comment_classifier(vocab_size,embedding_dim,hidden_size,num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    print(model)

    num_eporch=5

    for epoch in range(num_eporch):
        for i,(cmt,lbl) in enumerate(dataloader):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            #向前传播
            output = model(cmt)
            loss = criterion(output,lbl)

            #反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_eporch}],step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(),'E:\\workspacepython\\AiPremiumClass\\commetn_classifier.pth')
    torch.save(vocab,'E:\\workspacepython\\AiPremiumClass\\vocab.pth')
   
    
 




