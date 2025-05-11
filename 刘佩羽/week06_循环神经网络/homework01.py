# 导入必要的包
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter

# 数据及加载
dataset = fetch_olivetti_faces(data_home='./scikit_learn_data/fetch_olivetti_faces',
                                    shuffle=True, download_if_missing=True, return_X_y=False)

print(dataset.keys())

# 数据预处理
X = dataset.images
y = dataset.target

# 数据集划分
train_size = int(0.8 * len(X))
valid_size = int(0.1 * len(X))
test_size = int(0.1 * len(X))

X_train, X_valid, X_test = X[:train_size], X[train_size:train_size+valid_size], X[train_size+valid_size:]

y_train, y_valid, y_test = y[:train_size], y[train_size:train_size+valid_size], y[train_size+valid_size:]

print(' X_train.shape', X_train.shape, 'X_valid.shape', X_valid.shape, 'X_test.shape', X_test.shape)

# 数据集类型转换
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_valid = torch.tensor(y_valid, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 数据集封装
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 100

# 数据集迭代器
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size, shuffle=False)

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=1, rnn_type='RNN'):
        super(RNN, self).__init__()
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# 初始化RNN
input_size = 64
hidden_size = 100
num_layers = 2
num_classes = 40

# 1.RNN;  2.GRU;  3.LSTM;
model = RNN(input_size, hidden_size, num_layers, num_classes, rnn_type='LSTM')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# tensorboard 
writer = SummaryWriter()

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 64 == 0:
            print('Epoch [{}/{}],Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
            writer.add_scalar('train/loss_val', loss.item(), epoch*64 + i)
    # 验证模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Validation Accuracy : {} %'.format(100 * correct / total))
        writer.add_scalar('Validation/Accuracy', (100 * correct / total), epoch*64 + i)
writer.close()
# # 测试模型
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Test Accuracy : {} %'.format(100 * correct / total))

