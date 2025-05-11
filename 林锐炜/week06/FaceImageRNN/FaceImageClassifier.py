import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class FaceImageClassifier(nn.Module):

    def __init__(self,model_type,input_size,hidden_size,num_classes):
        super(FaceImageClassifier,self).__init__()
        
        self.bidirectional = False
        match model_type:
            case 'RNN':
                self.rnn = nn.RNN(input_size,hidden_size,batch_first=True,num_layers=5)
            case 'LSTM':
                self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=5)
            case 'GRU':
                self.rnn = nn.GRU(input_size,hidden_size,batch_first=True,num_layers=5)
            case 'BiLSTM':
                self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=True,num_layers=5)
                self.bidirectional = True
            case _:
                raise ValueError("Unsupported model type. Choose from 'LSTM', 'GRU', or 'RNN'.")
        
        # 根据双向标志调整全连接层输入维度
        fc_input_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size,num_classes)
    
    def forward(self, x):
        # 更新LSTM初始化状态
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.rnn.num_layers*(2 if self.bidirectional else 1),
                           x.size(0), self.rnn.hidden_size).to(device)
            c0 = torch.zeros_like(h0)
            outputs, _ = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.rnn.num_layers, x.size(0), 
                           self.rnn.hidden_size).to(device)
            outputs, _ = self.rnn(x, h0)
            
        out = self.fc(outputs[:, -1, :])
        return out
    
# 加载和预处理数据
def load_data():
    # 加载数据集
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # 将数据转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为3D张量
    X_train = X_train.view(-1, 64, 64)
    X_test = X_test.view(-1, 64, 64)
    
    return X_train, y_train, X_test, y_test

def train_model(model_type, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=16, learning_rate=0.001):
    input_size = 64
    hidden_size = 64
    num_classes = 40
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 实例化模型
    model = FaceImageClassifier(model_type=model_type,input_size=input_size,hidden_size=hidden_size,num_classes=num_classes)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(f"runs/face_image_rnn_{model_type}")
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if i % 100 == 0:
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
                print(f'Epoch training [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            writer.add_scalar('test accuracy', accuracy, epoch)
            print(f'Epoch test [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
    
    torch.save(model, f"face_image_rnn_{model_type}.pth")
    torch.save(model.state_dict(), f"face_image_rnn_{model_type}_params.pth")
    writer.close()
    return accuracy

def load_model(model_type):
    # 加载模型
    model = torch.load(f"face_image_rnn_{model_type}.pth")
    # 加载模型参数
    model = FaceImageClassifier(model_type=model_type)
    model.load_state_dict(torch.load(f"face_image_rnn_{model_type}_params.pth"))
    return model
    
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()

    # 比较不同模型
    models = ['RNN', 'LSTM', 'GRU', 'BiLSTM']
    results = {}

    for model_type in models:
        print(f"\nTraining {model_type}...")
        acc = train_model(model_type, X_train, y_train, X_test, y_test)
        results[model_type] = acc


    print("\nFinal Results:")
    for model, acc in results.items():
        print(f"{model}: {acc}")