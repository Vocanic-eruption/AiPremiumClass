import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. 数据向量化
max_len = 200  
def text_to_indices(tokens):
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens][:max_len]
    return indices + [word2idx['<PAD>']]*(max_len - len(indices)) 

X = [text_to_indices(tokens) for tokens in df['tokens']]
y = df['label'].values

# 2. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 3. 创建DataLoader
train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_data = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
val_loader = DataLoader(val_data, batch_size=64)

# 4. 定义LSTM模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 二分类
    
    def forward(self, x):
        x = self.embedding(x) 
        _, (hidden, _) = self.lstm(x) 
        return self.fc(hidden.squeeze(0))

model = TextClassifier(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练与评估函数
def train():
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 6. 训练循环
for epoch in range(10):
    loss = train()
    val_acc = evaluate(val_loader)
    print(f'Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}')

# 7. 测试集评估
test_acc = evaluate(DataLoader(TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test)), batch_size=64))
print(f'Final Test Accuracy: {test_acc:.4f}')
