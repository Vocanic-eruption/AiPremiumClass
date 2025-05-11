# -*- coding: utf-8 -*-
# @Date    : 2025/4/6 13:08
# @Author  : Lee
import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# 加载数据集
olivetti_face = fetch_olivetti_faces(
    data_home=r"C:\Users\liyang\AI&机器学习\八斗学院学习笔记\第四周_2025_3_16学习内容\face_data", shuffle=True)

# 将数据集分割数据和标签转化为 torch.tensor 格式
X = torch.tensor(olivetti_face.data, dtype=torch.float32)
y = torch.tensor(olivetti_face.target, dtype=torch.long)

# 将数据分割为训练集、验证集和测试集
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# 创建数据加载器
# 转换为 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#创建writer 路径
log_dir = 'runss'
writer = SummaryWriter(log_dir = log_dir)

# RNN 模型创建
class RNN_Classifiters(nn.Module):
    def __init__(self, hidden_size=100):
        super(RNN_Classifiters, self).__init__()
        self.Rnn = nn.RNN(
            input_size=64,  # 每行的像素数
            hidden_size=hidden_size,  # 隐藏层单元数
            num_layers=1,    # RNN层数
            bias=True,
            batch_first=True
        )
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size,40)

    def forward(self, x):
        # 调整输入维度以匹配RNN的期望输入 [batch_size, sequence_length, input_size]
        x = x.view(x.size(0), 64, 64)  # 将图像转换为 [batch_size, sequence_length, input_size]
        outputs, _ = self.Rnn(x)
        out = self.tanh(outputs[:, -1, :]) # 取序列的最后一个时间步的输出
        out = self.dropout(out)
        final = self.fc(out)
        return final

#创建LSTM模型
# RNN 模型创建
class LSTM_Classifiters(nn.Module):
    def __init__(self, hidden_size=100):
        super(LSTM_Classifiters, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=64,  # 每行的像素数
            hidden_size=hidden_size,  # 隐藏层单元数
            num_layers=1,    # RNN层数
            bias=True,
            batch_first=True
        )
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size,40)

    def forward(self, x):
        # 调整输入维度以匹配RNN的期望输入 [batch_size, sequence_length, input_size]
        x = x.view(x.size(0), 64, 64)  # 将图像转换为 [batch_size, sequence_length, input_size]
        outputs, _ = self.LSTM(x)
        out = self.tanh(outputs[:, -1, :]) # 取序列的最后一个时间步的输出
        out = self.dropout(out)
        final = self.fc(out)
        return final


# 训练模型
def train_model(models, train_loader, val_loader, num_epochs=10):
    for model in models:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        print(f'模型使用{model}')
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
                    writer.add_scalar('training_loss:', loss.item(), epoch * len(train_loader) + i)

        # 验证过程
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Acc: {acc}%")
        writer.add_scalar("test_acc", acc, epoch)
    return running_loss / len(train_loader), acc

model1 = RNN_Classifiters()
model2 = LSTM_Classifiters()
models = [model1,model2]
num_epochs=20

# 训练模型
train_model(models, train_loader, val_loader, num_epochs=10)
writer.close()

# # 测试模型
# def test_model(model, test_loader, criterion):
#     model.eval()  # 将模型设置为评估模式
#     total_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():  # 关闭梯度计算
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     avg_loss = total_loss / len(test_loader)
#     print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
#
# # 调用测试函数
# test_model(model, test_loader, criterion)