import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os

# 获取当前文件所在目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))



# 数据准备
class FaceDataset(Dataset):
    def __init__(self, images, labels, seq_len=64):
        self.images = images.reshape(-1, 64, 64)  # Olivetti图像尺寸64x64
        self.labels = labels
        self.seq_len = seq_len  # 将图像视为64个64维的序列

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 将图像转换为序列格式 (seq_len, feature_size)
        sequence = self.images[idx].reshape(self.seq_len, -1)
        return torch.FloatTensor(sequence), torch.LongTensor([self.labels[idx]])

# 模型定义
class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'birnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            hidden_size *= 2  # 双向RNN需要加倍隐藏层大小
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 训练配置
def train_model(model, train_loader, test_loader, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(os.path.join(current_dir, f'runs/face_rnn_{model_name}'))
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 记录训练损失
        writer.add_scalar(f'Training Loss/{model_name}', total_loss/len(train_loader), epoch)
        
        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        writer.add_scalar(f'Test Accuracy/{model_name}', accuracy, epoch)
    
    writer.close()
    return accuracy

# 主程序
if __name__ == '__main__':
    # 加载数据
    faces = fetch_olivetti_faces()
    X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)
    
    # 创建数据集
    train_dataset = FaceDataset(X_train, y_train)
    test_dataset = FaceDataset(X_test, y_test)
    
    # 参数设置
    batch_size = 64
    input_size = 64  # 每个时间步的特征维度
    hidden_size = 128
    num_classes = 40  # Olivetti有40个不同的人
    num_layers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练不同模型
    models = {
        'RNN': 'rnn',
        'LSTM': 'lstm',
        'GRU': 'gru',
        'BiRNN': 'birnn'
    }
    
    results = {}
    for name, rnn_type in models.items():
        print(f"Training {name}...")
        model = RNNModel(rnn_type, input_size, hidden_size, num_classes, num_layers).to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        acc = train_model(model, train_loader, test_loader, device, name)
        results[name] = acc
    
    print("\nFinal Results:")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}%")