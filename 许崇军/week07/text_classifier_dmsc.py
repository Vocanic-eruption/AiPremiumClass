import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import jieba
import pandas as pd
from collections import Counter
import sentencepiece as spm
import logging
jieba.setLogLevel(logging.INFO)


def jieba_tokenize(text):
    return jieba.lcut(text)

def sentencepiece_tokenize(text):
    sp = spm.SentencePieceProcessor()
    sp.Load('sentencepiece.model')
    return sp.EncodeAsPieces(text)

def load_data(file_path):
    df = pd.read_csv(file_path)
    data = []
    for index, row in df.iterrows():
        score = row['Star']
        if 1 <= score <= 2:
            label = 1
        elif 4 <= score <= 5:
            label = 0
        else:
            continue
        comment = row['Comment']
        data.append((comment, label))
    return data


def build_from_doc(doc, tokenize_func):
    vocab = set()
    for line in doc:
        words = tokenize_func(line[0])
        vocab.update(words)
    vocab = ['PAD', 'UNK'] + list(vocab)
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

class DoubanDataset(Dataset):
    def __init__(self, data, vocab, tokenize_func):
        self.data = data
        self.vocab = vocab
        self.tokenize_func = tokenize_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        comment, label = self.data[idx]
        words = self.tokenize_func(comment)
        comment_idx = [self.vocab.get(word, self.vocab['UNK']) for word in words]
        return torch.tensor(comment_idx), torch.tensor(label)

class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output

def convert_data(batch_data):
    comments, votes = [], []
    for comment, vote in batch_data:
        comments.append(comment)
        votes.append(vote)
    commt = pad_sequence(comments, batch_first=True, padding_value=0)
    labels = torch.tensor(votes)
    return commt, labels

def train_and_test(data, tokenize_func):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = build_from_doc(data, tokenize_func)
    print(f'使用 {tokenize_func.__name__} 分词，词汇表大小: {len(vocab)}')

    # 划分训练集和测试集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 创建数据集和数据加载器
    train_dataset = DoubanDataset(train_data, vocab, tokenize_func)
    test_dataset = DoubanDataset(test_data, vocab, tokenize_func)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=convert_data)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=convert_data)

    # 构建模型
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, num_classes)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (cmt, lbl) in enumerate(train_dataloader):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            # 前向传播
            outputs = model(cmt)
            loss = criterion(outputs, lbl)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for cmt, lbl in test_dataloader:
            cmt = cmt.to(device)
            lbl = lbl.to(device)
            outputs = model(cmt)
            _, predicted = torch.max(outputs.data, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()

    accuracy = 100 * correct / total
    print(f'使用 {tokenize_func.__name__} 分词，测试集准确率: {accuracy}%')
    return accuracy

if __name__ == '__main__':

    data = load_data('/kaggle/input/doubanmovieshortcomments/DMSC.csv')

    tokenize_functions = [jieba_tokenize, sentencepiece_tokenize]
    results = {}

    for func in tokenize_functions:
        accuracy = train_and_test(data, func)
        results[func.__name__] = accuracy

    print("\n不同分词工具的准确率对比:")
    for func_name, acc in results.items():
        print(f"{func_name}: {acc}%")

    # 不同分词工具的准确率对比:
    # jieba_tokenize: 92.6829268292683 %
    # sentencepiece_tokenize: 91.46341463414635 %