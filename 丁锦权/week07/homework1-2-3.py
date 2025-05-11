import pandas as pd
import re
import sentencepiece as spm

from sklearn.feature_extraction.text import TfidfVectorizer

# 加载数据
data = pd.read_csv("DMSC.csv",encoding='utf-8')

# 筛选需要的列
data = data[['Comment', 'Star']]

# 定义情感标签
data['Sentiment'] = data['Star'].apply(lambda x: 1 if x <= 2 else 0)

# 清洗文本
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['Cleaned_Comment'] = data['Comment'].apply(clean_text)

# 将清洗后的评论保存到文件用于训练SentencePiece模型
with open('cleaned_comments.txt', 'w', encoding='utf-8') as f:
    for comment in data['Cleaned_Comment']:
        f.write(comment + '\n')

# 训练SentencePiece模型
spm.SentencePieceTrainer.train(
    input='cleaned_comments.txt',
    model_prefix='douban_comment',
    vocab_size=8000,
    character_coverage=0.9995,
    model_type='bpe'
)

# 加载训练好的模型
sp = spm.SentencePieceProcessor()
sp.load('douban_comment.model')

# 构建词典
vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

# 使用训练好的模型对评论进行分词和编码
data['Encoded_Comment'] = data['Cleaned_Comment'].apply(lambda x: sp.encode(x, out_type=int))


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 定义RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 超参数
input_size = 128  # 词向量维度
hidden_size = 128  # RNN隐藏层维度
num_layers = 2  # RNN层数
num_classes = 2  # 分类数（positive和negative）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['Encoded_Comment'], data['Sentiment'], test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
X_train = [torch.tensor(comment).unsqueeze(0).float() for comment in X_train]
X_test = [torch.tensor(comment).unsqueeze(0).float() for comment in X_test]
y_train = torch.tensor(y_train.tolist())
y_test = torch.tensor(y_test.tolist())

# 定义模型、损失函数和优化器
model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        model.train()
        comment = X_train[i].unsqueeze(0).float()  # 增加批次维度
        label = y_train[i].unsqueeze(0)  # 增加批次维度

        # 前向传播
        outputs = model(comment)
        loss = criterion(outputs, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = []
    for comment in X_test:
        comment = comment.unsqueeze(0).float()  # 增加批次维度
        output = model(comment)
        _, predicted = torch.max(output, 1)
        y_pred.append(predicted.item())

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

import jieba
# 使用jieba分词
data['Cut_Comment_jieba'] = data['Cleaned_Comment'].apply(lambda x: ' '.join(jieba.cut(x)))




vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec_jieba = vectorizer.fit_transform(data['Cut_Comment_jieba'])

# 训练和评估模型（以jieba为例）
model.fit(X_train_vec_jieba, data['Sentiment'])
y_pred_jieba = model.predict(X_train_vec_jieba)
print(f'Accuracy with jieba: {accuracy_score(y_test, y_pred_jieba)}')