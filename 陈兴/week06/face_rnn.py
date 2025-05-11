import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from datetime import datetime

# 1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。
#  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html

# ======== 1. Load Dataset ========
faces = fetch_olivetti_faces()
X = faces.images  # (400, 64, 64)
y = faces.target  # (400,)

# Normalize and reshape for RNN input (batch, seq_len, input_size)
X = X.astype(np.float32)
X = X.reshape(-1, 64, 64) # 为了代码的可读性和一致性, 这里并没有改变X的形状

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ======== 2. Dataset & Dataloader ========
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FaceDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(FaceDataset(X_test, y_test), batch_size=32)

# ======== 3. Define RNN Classifier ========
class RNNClassifier(nn.Module):
    def __init__(self, rnn_type='LSTM', input_size=64, hidden_size=128, num_classes=40, bidirectional=False):
        super().__init__()
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size,
                           batch_first=True, bidirectional=bidirectional)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, num_classes)

    def forward(self, x):
        # x: (batch, seq_len=64, input_size=64)
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]  # take last timestep output
        return self.fc(last)

# ======== 4. Train and Evaluate ========
def train_model(rnn_type='LSTM', bidirectional=False, log_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(rnn_type=rnn_type, bidirectional=bidirectional).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(20):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                pred = outputs.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        test_acc = correct / total
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f"Epoch {epoch}: Train Acc: {acc:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()

# ======== 5. Run experiments ========
os.makedirs("./陈兴/week06/runs", exist_ok=True)

configs = [
    ('RNN', False),
    ('GRU', False),
    ('LSTM', False),
    ('LSTM', True),  # BiLSTM
]

for rnn_type, bi in configs:
    logdir = f"./陈兴/week06/runs/{rnn_type}{'_bi' if bi else ''}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTraining {rnn_type}, Bidirectional={bi}")
    train_model(rnn_type=rnn_type, bidirectional=bi, log_dir=logdir)
