#使用RNN实现一个天气预测模型，
# 能预测1天和连续5天的最高气温。
# 要求使用tensorboard，提交代码及run目录和可视化截图。
#数据集：URL_ADDRESS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler

# 加载数据集,look_back过去xx天，future_steps未来一天
def load_data(look_back=30, future_steps=1):
    # 加载数据集
    data = pd.read_csv('train.csv',low_memory=False)
    #  取最高温 数据预清洗,去除无效值NaN
    data = data[['Date', 'MaxTemp']].dropna()
    # 归一化最高气温
    scaler = MinMaxScaler(feature_range=(0,1))
    data['MaxTemp'] = scaler.fit_transform(data['MaxTemp'].values.reshape(-1,1))
    # 日期转换
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # 数据预处理, 使用过去N天的数据预测未来的气温
    def create_dataset(data, look_back, fulture_steps): # look_back 参数决定了我们使用多少个历史时间步的数据来预测下一个时间步的数据
        X, Y = [], []
        for i in range(len(data) - look_back - fulture_steps):
            X.append(data[i:i + look_back,0])
            Y.append(data[i + look_back: i + look_back + fulture_steps,0])
        return np.array(X), np.array(Y)

    # 设置时间步, 使用过去7天的气温数据预测未来气温
    look_back = 7
    dataset = data.values
    X, Y = create_dataset(dataset, look_back, future_steps)
    X = np.array(X)
    Y = np.array(Y) 
    # 拆分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]  
    # 转换成张量 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # 创建DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10,shuffle=False)
    return train_loader,test_loader


# 定义超参数
learning_rate = 0.001
num_epochs = 30
hidden_size = 7
num_layers = 1
num_classes = 1 # 预测最高气温
input_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

# 定义模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
        
        
# 定义训练函数
def train_func(model, train_loader, model_name="LSTM"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=f"{model_name}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad() # 梯度清零
            outputs = model(X_batch) # 前向传播
            loss = criterion(outputs, y_batch) # 计算损失
            loss.backward() # 反向传播
            optimizer.step()  # 更新参数
            total_loss +=loss.item()
            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            
        # 计算平均损失
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}")
        writer.add_scalar(f"{model_name}/Train_Loss", avg_train_loss, epoch)
        
# 训练
train_loader,test_loader = load_data()
model = RNN(input_size, hidden_size, num_layers, num_classes)
train_func(model, train_loader, model_name="LSTM")

#测试
print("Testing...")
model = RNN(input_size, hidden_size, num_layers, num_classes)
train_func(model, test_loader, model_name="LSTM-test")
    
    
