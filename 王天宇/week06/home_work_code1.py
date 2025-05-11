#1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，
# 如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，
# 提交代码及run目录和可视化截图。


from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# 设置超参数
learning_rate = 0.001
num_epochs = 1000
input_size = 4096  # 每个图像的特征数
hidden_size = 128  # LSTM 隐藏层的大小
num_layers = 1     # LSTM 层数
num_classes = 40   # 类别数
sequence_length = 1  # 序列长度，这里设置为1，因为每个输入是一个图像

# 加载数据集
data = fetch_olivetti_faces(data_home='./', shuffle=True)
X = data.data
y = data.target
print(X.shape) # (400, 4096)
print(y.shape) # (400,)

# 可视化部分数据
# plt.figure(figsize=(10, 10))
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(X[i].reshape(64, 64), cmap='gray')
#     plt.axis('off')
# plt.show()

# 数据预处理
X = X.reshape(400, 4096)
X_train = X[:300]
X_test = X[300:]
y_train = y[:300]
y_test = y[300:]
# print(X_train.shape) # (300, 4096)
# print(X_test.shape) # (100, 4096)
# print(y_train.shape) # (300,)
# print(y_test.shape) # (100,)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # 设置隐藏层的大小
        self.num_layers = num_layers # 设置 LSTM 层数
        # 定义 LSTM 层，输入维度为 input_size，隐藏层维度为 hidden_size，层数为 num_layers，batch_first=True 表示输入数据的第一个维度是 batch_size
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，输入维度为 hidden_size，输出维度为 num_classes（类别数）
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 通过 LSTM 层处理输入 x，返回输出 out 和隐藏状态 _（这里不需要隐藏状态，所以用 _ 忽略）
        out, _ = self.lstm(x)
        # 取 LSTM 输出的最后一个时间步的输出，形状为 (batch_size, hidden_size)，并通过全连接层得到最终输出
        out = self.fc(out[:, -1, :])
        return out

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard 记录
writer = SummaryWriter()

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            writer.add_scalar('training loss-GRU', loss.item(), epoch * total_step + i)

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 100 test images: {} %'.format(100 * correct / total))
    
# 关闭 TensorBoard writer
writer.close()