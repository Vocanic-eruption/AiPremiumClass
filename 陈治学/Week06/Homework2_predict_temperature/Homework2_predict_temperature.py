import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
import os 
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体和符号显示
try:
    rcParams['font.sans-serif'] = ['SimHei']  # 使用系统自带黑体
    rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
except:
    # 如果系统没有默认中文字体，使用指定字体文件（需要确保文件存在）
    zh_font = {'family': 'Microsoft YaHei', 'weight': 'normal', 'size': 10}
    rcParams['font.sans-serif'] = [zh_font['family']]
    rcParams['axes.unicode_minus'] = False
    plt.rc('font', **zh_font)
    
# 1. 数据预处理
def load_data():
    # 从Kaggle加载数据集（请确保文件已下载到本地）
    df = pd.read_csv(r'F:/NLP算法课程/正式课/0329/climate_data/Summary of Weather.csv')
    data = df['MaxTemp'].values.astype('float32')
    data = data.reshape(-1, 1)
    
    # 处理缺失值
    data = np.where(np.isnan(data), np.nanmean(data), data)
    
    # 归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler

# 2. 创建数据集
def create_dataset(data, look_back=7, predict_steps=5):  # 修改预测步数
    X, y = [], []
    # 调整循环条件确保数据对齐
    for i in range(len(data) - look_back - predict_steps):
        X.append(data[i:(i+look_back), 0])
        y.append(data[(i+look_back):(i+look_back+predict_steps), 0])
    return np.array(X), np.array(y)

# 3. 模型构建
class TemperaturePredictor(nn.Module):
    # def __init__(self, input_size=1, hidden_size=64, output_size=1):
    #     super(TemperaturePredictor, self).__init__()
    #     self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
    #     self.fc = nn.Linear(hidden_size, output_size)
    def __init__(self, input_size=1, hidden_size=64, output_size=5):  # 输出5个预测值
        super(TemperaturePredictor, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 直接输出5个预测值
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

def build_model(predict_steps):
    model = TemperaturePredictor(output_size=predict_steps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer

# 4. 训练配置
def train_model(model, criterion, optimizer, X_train, y_train, log_dir):
    writer = SummaryWriter(log_dir)
    
    # 划分验证集（时间序列数据按顺序分割）
    split = int(0.8 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_train, y_train = X_train[:split], y_train[:split]
    
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), 
                            batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val)), 
                          batch_size=32)

    for epoch in range(50):
        # 训练阶段
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs
            # inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # inputs = inputs.unsqueeze(-1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        # 记录指标
        writer.add_scalars('Loss', {
            'train': train_loss/len(train_loader),
            'val': val_loss/len(val_loader)
        }, epoch)
        
        # 每10个epoch记录一次预测样例
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 对整个验证集进行预测
                val_inputs = torch.Tensor(X_val)
                val_preds = model(val_inputs)
                
                # 创建可视化图表
                fig = plt.figure(figsize=(15, 6))
                # 绘制最后30个样本的真实值和预测值
                plt.plot(y_val[-30:].flatten(), label='Actual_value', marker='o')
                plt.plot(val_preds.numpy()[-30:].flatten(), label='Predicted_value', marker='x')
                plt.title(f'Epoch {epoch} 验证集预测对比')
                plt.legend()
                writer.add_figure('validation/predictions', fig, epoch)
                plt.close()
    
    writer.close()
    return model

# 5. 预测函数
def predict_temperature(model, data, scaler, predict_steps=5):
    model.eval()
    look_back = 7
    with torch.no_grad():
        true_value_index = len(data) - predict_steps
        last_data = data[-look_back:]
        last_data = torch.Tensor(last_data).unsqueeze(0)  # 形状 [1, 7, 1]
        prediction = model(last_data)  # 输出形状 [1, 5]
        
        # 获取5天的真实值
        true_value = scaler.inverse_transform(data[true_value_index:true_value_index+predict_steps])
    return scaler.inverse_transform(prediction.numpy()), true_value

# 主程序修改部分
def main():
    PREDICT_STEPS = 5
    LOG_DIR = r'F:/temp/tensorboard2/predict5_step'
    writer = SummaryWriter(log_dir=LOG_DIR)

    data, scaler = load_data()
    X, y = create_dataset(data, predict_steps=PREDICT_STEPS)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 添加特征维度 [样本数, 时间步长, 特征数]
    # 构建模型
    model, criterion, optimizer = build_model(PREDICT_STEPS)
    
    # 训练模型
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR,exist_ok=True)

    trained_model = train_model(model, criterion, optimizer, X, y, LOG_DIR)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'temperature_prediction.pth')
    
    # 示例预测
    # 获取最后31天的真实值（逆归一化后）
    last_31_days = scaler.inverse_transform(data[-31:])
    
    # 生成连续预测（每天基于前7天预测）
    predictions = []
    for day in range(30, 31):
        input_seq = data[day-7:day]  # 原始形状 [7, 1]
        input_tensor = torch.Tensor(input_seq)
        input_tensor = input_tensor.unsqueeze(0)  # 正确3D形状 [1, 7, 1]
        pred = model(input_tensor)
        predictions.append(scaler.inverse_transform(pred.detach().numpy()))
    
    # 修改可视化部分
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 绘制31天真实值
    ax.plot(range(31), last_31_days.flatten(), label='真实温度', marker='o')
    
    # 绘制5日预测值（从第31天开始）
    pred_days = range(30, 35) if PREDICT_STEPS ==5 else [30]
    ax.scatter(pred_days, predictions[-1].flatten(), color='red', label='预测温度', marker='X', s=100)
    
    # 添加标签（调整索引范围）
    for i in range(31):
        ax.annotate(f'{last_31_days[i][0]:.1f}°C', (i, last_31_days[i][0]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    # 调整预测值标签范围
    for idx, val in enumerate(predictions[-1].flatten(), start=31):
        ax.annotate(f'{val:.1f}°C', (idx-1, val),  # x轴从30开始对应第31天
                   textcoords="offset points", xytext=(0,15), ha='center', color='red')
    
    ax.set_title(f'连续31天温度趋势与未来{PREDICT_STEPS}天预测')
    ax.set_xlabel('天数')
    ax.set_ylabel('温度 (°C)')
    ax.legend()
    
    # 添加数据标签
    for i, (true, pred) in enumerate(zip(last_31_days, [*predictions, None])):
        if i == 30:
            ax.annotate(f'{true[0]:.1f}°C', (i, true[0]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(f'{predictions[-1][0][0]:.1f}°C', 
                (30, predictions[-1][0][0]), 
                textcoords="offset points", 
                xytext=(0,15), 
                ha='center',
                color='red')

    writer.add_figure('31_days_comparison', fig)
    plt.savefig('31_days_forecast.png')
    plt.close()

if __name__ == "__main__":
    main()

# tensorboard --logdir=F:/temp/tensorboard2/predict1_step