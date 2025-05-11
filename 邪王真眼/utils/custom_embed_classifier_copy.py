import torch
import torch.nn as nn
import torch.optim as optim

# 数据准备
# 简单的文本数据集
texts = ["I love this movie", 
         "This movie is terrible", 
         "Great film", 
         "Awful experience"]
labels = [1, 0, 1, 0]

# 构建词汇表
vocab = set()
for text in texts:
    for word in text.split():
        vocab.add(word)

vocab = sorted(vocab)
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# 将文本转换为索引序列
texts_idx = []
for text in texts:
    idx_seq = [word_to_idx[word] for word in text.split()]
    texts_idx.append(torch.tensor(idx_seq))

labels = torch.tensor(labels)

# 手动实现 Embedding 层
class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MyEmbedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_ids):
        return self.embedding_matrix[input_ids]

# 定义 RNN+FC 模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = MyEmbedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output

# 模型参数
embedding_dim = 10
hidden_size = 20
num_classes = 2

# 初始化模型
model = TextClassifier(vocab_size, embedding_dim, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(texts_idx)):
        input_ids = texts_idx[i]
        label = labels[i]

        # 前向传播
        output = model(input_ids)
        loss = criterion(output, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(texts_idx):.4f}')

# 测试模型
test_texts = ["I love this movie", "This film is terrible"]
test_texts_idx = []
for text in test_texts:
    idx_seq = [word_to_idx[word] for word in text.split()]
    test_texts_idx.append(torch.tensor(idx_seq))

# 将测试数据转换为张量
test_texts_idx = [torch.tensor(idx_seq).unsqueeze(0) for idx_seq in test_texts_idx]
test_texts_idx = torch.cat(test_texts_idx, dim=0)
# 模型评估
model.eval()
with torch.no_grad():
    for input_ids in test_texts_idx:
        output = model(input_ids)
        _, predicted = torch.max(output, 0)
        print(f'Input: {input_ids}, Predicted: {predicted.item()}')
    