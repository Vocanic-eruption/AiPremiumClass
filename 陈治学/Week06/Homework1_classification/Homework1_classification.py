import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# 数据准备
data = fetch_olivetti_faces()
X = torch.FloatTensor(data.images).unsqueeze(1)  # (400, 1, 64, 64)
y = torch.LongTensor(data.target)  

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# 定义模型基类
class FaceRNN(nn.Module):
    def __init__(self, rnn_type='RNN'):
        super().__init__()
        self.norm = nn.LayerNorm(64)  # 输入归一化
        self.dropout = nn.Dropout(0.3)  # 新增dropout层
        
        # 统一配置不同RNN类型
        rnn_params = {
            'input_size': 64,
            'hidden_size': 256,
            'num_layers': 2,
            'batch_first': True,
            'dropout': 0.3 if 2 > 1 else 0
        }
        
        # 动态创建不同RNN类型
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(**rnn_params)
            fc_input = 256
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(**rnn_params)
            fc_input = 256
        else:  # 普通RNN
            self.rnn = nn.RNN(**rnn_params)
            fc_input = 256
            
        self.fc = nn.Linear(fc_input, 40)

    def forward(self, x):
        x = x.squeeze(1)    # (batch, 64, 64)
        x = self.norm(x)    # 归一化
        x, _ = self.rnn(x)  # 统一处理所有RNN类型
        x = x[:, -1, :]     # 取最后时间步
        x = self.dropout(x) # dropout
        return self.fc(x)

# 训练函数
def train_model(model, name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    log_dir = r'F:\temp\tensorboard' # 指定日志保存路径
    writer = SummaryWriter(log_dir=f'{log_dir}/{name}')

    
    for epoch in range(50):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('Accuracy', acc, epoch)
    
    writer.close()

# 训练不同模型
for rnn_type in ['RNN', 'LSTM', 'GRU']:
    model = FaceRNN(rnn_type)
    train_model(model, f'FaceRecognition_{rnn_type}')

# 启动TensorBoard (在终端运行)
# tensorboard --logdir=F:\temp\tensorboard --port=6006
