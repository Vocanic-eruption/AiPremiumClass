import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载Olivetti人脸数据集
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.images
y = faces.target

# 数据预处理
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # (400, 64, 64)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化像素值到[0, 1]范围
scaler = MinMaxScaler()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 重塑数据以适应RNN输入: (样本数, 时间步长, 特征数)
# 我们将每行像素视为一个时间步长
X_train = X_train.reshape(X_train.shape[0], 64, 64)
X_test = X_test.reshape(X_test.shape[0], 64, 64)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建TensorBoard的日志目录
run_dir = "run"
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# 定义RNN模型类
class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size=64, hidden_size=128, num_classes=40):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        
        # 根据指定的RNN类型选择不同的RNN层
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x的形状: (batch_size, seq_length, input_size)
        
        # RNN的前向传播
        if self.rnn_type == "LSTM":
            output, (hidden, _) = self.rnn(x)
        else:  # GRU或RNN
            output, hidden = self.rnn(x)
        
        if self.rnn_type == "LSTM" or self.rnn_type == "GRU":
            out = hidden[-1, :, :]
        else:  # 简单RNN
            out = hidden[-1, :, :]
        
        # 通过全连接层进行分类
        out = self.fc(out)
        return out

# 训练模型函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, writer, rnn_type):
    # 移动模型到设备
    model.to(device)
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练精度
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # 评估模式
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算测试精度
        test_accuracy = 100 * correct / total
        test_loss = test_loss / len(test_loader)
        
        # 记录到TensorBoard
        writer.add_scalar(f'{rnn_type}/Loss/train', train_loss, epoch)
        writer.add_scalar(f'{rnn_type}/Loss/test', test_loss, epoch)
        writer.add_scalar(f'{rnn_type}/Accuracy/train', train_accuracy, epoch)
        writer.add_scalar(f'{rnn_type}/Accuracy/test', test_accuracy, epoch)
        
        # 输出训练信息
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(run_dir, f"{rnn_type}_model.pth"))
    return model

# 训练所有模型
def train_all_models(rnn_types, epochs=50):
    for rnn_type in rnn_types:
        print(f"\n开始训练 {rnn_type} 模型...")
        
        # 创建模型
        model = RNNModel(rnn_type=rnn_type)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 创建TensorBoard writer
        log_dir = os.path.join(run_dir, f"{rnn_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir=log_dir)
        
        # 训练模型
        train_model(model, train_loader, test_loader, criterion, optimizer, epochs, writer, rnn_type)
        
        # 关闭TensorBoard writer
        writer.close()

# 运行实验，对比不同RNN结构
rnn_types = ["LSTM", "GRU", "RNN"]
train_all_models(rnn_types, epochs=50)

print("\n实验已完成。请查看'run'目录中的TensorBoard日志以获取可视化结果。")
print("要查看TensorBoard，请运行: tensorboard --logdir=run")
