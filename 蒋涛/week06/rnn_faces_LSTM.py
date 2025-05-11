import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 配置参数
class Config:
    batch_size = 32
    input_size = 64      # 每行像素作为特征
    seq_len = 64         # 图像高度作为序列长度
    hidden_size = 128
    num_classes = 40
    num_layers = 3
    lr = 0.001
    epochs = 50

class FaceRNN(nn.Module):
    def __init__(self):
        super(FaceRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=Config.input_size,
            hidden_size=Config.hidden_size,
            num_layers=Config.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(Config.hidden_size, Config.num_classes)

    def forward(self, x):
        x = x.view(-1, Config.seq_len, Config.input_size)  # [batch, 64, 64]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def load_faces():
    data = fetch_olivetti_faces()
    X = data.data.reshape(-1, 64, 64).astype(np.float32)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_dataset, test_dataset

def train_face():
    writer = SummaryWriter(log_dir='runs/face_lstm')
    
    # 准备数据
    train_set, test_set = load_faces()
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 验证
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()

        acc = 100 * correct / total
        writer.add_scalars('Loss', {'train': loss.item()}, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        print(f'Epoch [{epoch+1}/{Config.epochs}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')

    writer.close()

if __name__ == '__main__':
    train_face()