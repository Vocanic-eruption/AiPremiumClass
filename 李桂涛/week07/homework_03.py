import torch.nn as nn
import torch
import torch.optim as optim
import kagglehub
import os
import pandas as pd
import numpy as np
import thulac
from sklearn.model_selection import train_test_split


thu = thulac.thulac(seg_only=True)  # seg_only模式只进行分词

# Download latest version
path = 'C:/Users/ligt/.cache/kagglehub/datasets/utmhikari/doubanmovieshortcomments/versions/7/'
print("Path to dataset files:", path)

data_path = os.path.join(path,"DMSC.csv")
data = pd.read_csv(data_path)
data = data[['Star','Comment']]

# 星数转换函数保持不变
def convert_star(Star):
    if Star in [1,2]:
        return 1
    elif Star in [4,5]:
        return 0
    else:
        return -1
data['Star'] = data['Star'].apply(convert_star)
data = data[data['Star'] != -1]

# 加载停用词
stop_set = set()
def load_stop():
    with open('C:/Users/ligt/bd_AI/w07/stop_words.txt','r',encoding='utf-8') as f:
        return {line.strip() for line in f}
stop_w = load_stop()

# 构建词汇表 使用THULAC分词
vocal = set()
for comment in data['Comment']:
    # 使用THULAC进行分词
    words = thu.cut(comment, text=True).split()  # text=True返回空格分隔的字符串
    words = [word for word in words if word not in stop_w]
    vocal.update(words)

vocal = sorted(vocal)
vocal_size = len(vocal)
print(f"词汇表大小：{vocal_size}")

# 创建词到索引的映射
word2idx = {word:i for i, word in enumerate(vocal)}

# 将文本转换为索引序列（修改3：THULAC分词处理）
text_idx = []
for ti in data['Comment']:
    words = thu.cut(ti, text=True).split()  # THULAC分词
    words = [w for w in words if w not in stop_w]
    idx_seq = [word2idx[w] for w in words if w in word2idx]
    
    # 处理空序列
    if len(idx_seq) == 0:
        idx_seq = [0]  # 使用第一个索引作为填充
    text_idx.append(torch.tensor(idx_seq))

text_label = torch.tensor(data['Star'].values)
x_train, x_val, y_train, y_val = train_test_split(text_idx, text_label, test_size=0.2, random_state=42)

# 添加序列填充,因为评论长短不一
from torch.nn.utils.rnn import pad_sequence

# 填充序列到相同长度
def pad_collate(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, labels

# 创建DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = [(seq, y_train[i]) for i, seq in enumerate(x_train)]
val_dataset = [(seq, y_val[i]) for i, seq in enumerate(x_val)]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=pad_collate)

# 模型
class ImprovedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)  #注意时双向的
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # 取最后一个时间步
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型参数
embedding_dim = 128
hidden_size = 256
num_classes = 2
LR = 1e-3
num_epochs = 10

model = ImprovedModel(
    vocab_size=len(vocal),
    embed_dim=embedding_dim,
    hidden_dim=hidden_size,
    num_classes=num_classes
).to(device)

# 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练函数
def train_func(model, criterion, optimizer, train_loader, val_loader, epochs):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 验证集评估
        val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {total_loss/len(train_loader):.3f} | Acc: {correct/total:.3f}')
        print(f'Val Acc: {val_acc:.3f}\n')

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if __name__ == "__main__":
    train_func(model, criterion, optimizer, train_loader, val_loader, num_epochs)