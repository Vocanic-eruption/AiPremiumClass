"""
Olivetti人脸数据集分类
对比 SimpleRNN/LSTM/GRU/BiRNN 性能
使用tensoboard记录训练
"""


from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# 🔹 初始化 TensorBoard 记录器（用于可视化训练过程）
writer = SummaryWriter(log_dir="./党金虎/week06/runs/olivetti_model")


# 1、数据准备 ======================
print("📌 加载 Olivetti Faces 数据集...")
# 加载人脸数据集,400张64x64, 40个人
data = fetch_olivetti_faces(data_home='./党金虎/week06/scikit_learn_data')
X = data.images  # (400, 64, 64)
y = data.target  # (400,) 

# 数据预处理
X = X[:, :, :, np.newaxis] # 增加通道纬度 (400, 64, 64)  →  (400, 64, 64, 1)
# 转 pytorch 张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 划分数据集 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 80%训练集，20%测试集 

train_dataset = TensorDataset(X_train, y_train) # 将训练数据集转换为张量
test_dataset = TensorDataset(X_test, y_test) # 将测试数据集转换为张量

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # 将训练数据集转换为加载器 32个样本一组 shuffle打乱顺序
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False) # 将测试数据集转换为加载器 32个样本一组 不打乱顺序


# 2、构建不同RNN模型 ======================
# (1) 普通RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) # RNN 层
        self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层（输出分类结果）

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])  # 适配 RNN 输入格式 (batch, seq, features)
        out, _ = self.rnn(x) # RNN前向传播
        out = self.fc(out[:, -1, :]) # 取最后一个时间步的输出
        return out

# (2)  LSTM（长短时记忆网络）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # LSTM 层
        self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层（输出分类结果）

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])  
        out, _ = self.lstm(x) # 前向传播
        out = self.fc(out[:,  -1, :]) # 取最后一个时间步的输出
        return out
    
# (3) GRU（门控循环单元）
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True) # RGRUNN 层
        self.fc = nn.Linear(hidden_size, num_classes)  # 全连接层（输出分类结果）

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1]) 
        out, _ = self.gru(x) # 前向传播
        out = self.fc(out[:, -1,:]) # 取最后一个时间步的输出
        return out
    
# (4) BiRNN（双向 RNN，使用双向 LSTM）
class BiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiRNNModel, self).__init__()
        self.birnn = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True) # 双向 LSTM
        self.fc = nn.Linear(hidden_size * 2, num_classes)   # 由于是双向 LSTM，需要 *2

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])  # 
        out, _ = self.birnn(x) # 前向传播
        out = self.fc(out[:,  -1, :]) # 取最后一个时间步的输出
        return out
    
# 3、训练评估函数 ======================

def train_and_evaluate(model, model_name, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器

    for epoch in range(num_epochs):
        model.train()  # 进入训练模式
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() # 梯度清零
            outputs = model(X_batch) # 前向传播
            loss = criterion(outputs, y_batch) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            
            total_loss +=loss.item()
            _, predicted = torch.max(outputs, 1)
            correct +=(predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
        train_acc = correct /total # 计算准确率
        writer.add_scalar(f"{model_name}/Loss", total_loss / len(test_loader), epoch)
        writer.add_scalar(f"{model_name}/Accuracy", train_acc, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | {model_name}: Loss={total_loss:.4f}, Accuracy={train_acc:.4f}")
    print(f"✅ {model_name} 训练完成！\n")


# 4 训练所有模型并记录到 TensorBoard
models = {
    "RNN": RNNModel(input_size=1, hidden_size=128, num_classes=40),
    "LSTM": LSTMModel(input_size=1, hidden_size=128, num_classes=40),
    "GRU": GRUModel(input_size=1, hidden_size=128, num_classes=40),
    "BiRNN": BiRNNModel(input_size=1, hidden_size=128, num_classes=40),
}

for name, model in models.items():
    train_and_evaluate(model, name)

writer.close()  # 关闭 TensorBoard 记录器

print("\n🎯 运行以下命令以查看 TensorBoard 结果：")
print("tensorboard --logdir=./党金虎/week06/runs/olivetti_model")

##################本地运行太耗时间,采用kaggle 52s · GPU P100 ######################################
# https://www.kaggle.com/code/zfy681/notebookcab263a10a/edit