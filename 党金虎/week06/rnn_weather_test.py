'''
使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温
'''


import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 加载数据==========================================
def load_and_preprocess_data(csv_path,look_back=7, future_steps=1):
    """
    读取天气数据，预处理并返回 DataLoader
    :param csv_path: CSV 文件路径
    :param look_back: 用过去N天预测未来
    :param future_steps: 预测未来N天
    :return: 训练集 & 测试集 DataLoader
    """
    data = pd.read_csv(csv_path,low_memory=False)
 
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
        X, y = [], []
        for i in range(len(data) - look_back - fulture_steps):
            X.append(data[i:i + look_back,0])
            y.append(data[i + look_back: i + look_back + fulture_steps,0])
        return np.array(X), np.array(y)

    # 设置时间步, 使用过去7天的气温数据预测未来气温
    look_back = 7
    dataset = data.values
    X, y = create_dataset(dataset, look_back, future_steps)

    # 数据拆分成训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换成张量 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 创建DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32,shuffle=False)
    return train_loader, test_loader, scaler



# 2. 构建LSTM训练模型==========================================
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(WeatherRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # out.shape: [batch, seq_len, hidden_size]
        return self.fc(out[:, -1, :])  # 只取最后一个时间步



# 3. 训练模型 & 评估训练, 损失函数设为MSE（均方误差）  Adam 优化器
def train_and_evaluate(model, train_loader, test_loader,epochs=100,lr=0.001,model_name="LSTM"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    writer = SummaryWriter(log_dir=f"./党金虎/week06/runs/weather_model/{model_name}")


    for epoch in range(epochs):
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
            
        # 计算平均损失
        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar(f"{model_name}/Train_Loss", avg_train_loss, epoch)
  
        # 评估
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs,y_batch)
                total_loss += loss.item()
        avg_test_loss = total_loss / len(test_loader)
        writer.add_scalar(f"{model_name}/Test_Loss", avg_test_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train_Loss: {avg_train_loss:.4f}, Test_Loss: {avg_test_loss:.4f} ")
    writer.close()
    

# 4.执行==========================================
def run_experiment(csv_path, future_steps,epochs=100):
    """
    运行天气预测实验
    :param csv_path: 数据集路径
    :param future_steps: 预测未来N天
    :param epochs: 训练轮数
    """
    print(f"📌 正在训练 {future_steps} 天预测模型...")
    train_loader, test_loader, _ = load_and_preprocess_data(csv_path,future_steps=future_steps)
    model = WeatherRNN(output_size = future_steps)
    train_and_evaluate(model,train_loader,test_loader,epochs,model_name=f"LSTM_{future_steps}_Days")


# 5.测试==========================================
csv_path ="./党金虎/week06/weather_tem_data/Summary of Weather.csv"
run_experiment(csv_path,future_steps=1,epochs=10) # 1天预测
run_experiment(csv_path,future_steps=5,epochs=10) # 5天预测

print("\n🎯 运行以下命令以查看 TensorBoard 结果：")
print("tensorboard --logdir=./党金虎/week06/runs/weather_model")