import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=5):
        """
        初始化天气预测RNN模型
        
        参数:
            input_size: 输入特征维度 (默认为1，即只使用最高温度)
            hidden_size: RNN隐藏层大小 (默认为64)
            output_size: 输出维度 (默认为5，预测未来5天)
        """
        super(WeatherRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True  # 设置输入格式为[batch, seq, feature]
        )
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层，将RNN输出映射到预测天数
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入数据，形状为[batch, times, features]
            
        返回:
            模型预测的未来温度值
        """
        # 输入x形状: [batch, times, features]
        outputs, h_n = self.rnn(x)  # RNN的所有输出
        # 使用最后一个时间步的输出
        out = self.fc(outputs[:,-1,:])
        return out

def load_weather_data(data_path, seq_length=14, pred_days=5):
    """
    加载和处理天气数据
    
    参数:
        data_path: 数据文件路径
        seq_length: 输入序列长度，即用多少天的历史数据作为特征 (默认14天)
        pred_days: 预测天数 (默认5天)
        
    返回:
        dataset: PyTorch数据集
        scaler: 用于反归一化预测结果的缩放器
    """
    # 使用提供的路径加载数据
    data = pd.read_csv(data_path, parse_dates=['Date'])
    temp = data['MaxTemp'].values.reshape(-1, 1)  # 提取最高温度并重塑为列向量
    scaler = MinMaxScaler()  # 创建归一化器
    scaled = scaler.fit_transform(temp)  # 归一化温度数据
    
    X, y = [], []
    # 构建滑动窗口数据
    for i in range(len(scaled)-seq_length-pred_days+1):
        X.append(scaled[i:i+seq_length])  # 输入序列
        y.append(scaled[i+seq_length:i+seq_length+pred_days].reshape(-1))  # 目标序列
    
    X = np.array(X).astype(np.float32)  # 形状: [n, 14, 1]
    y = np.array(y).astype(np.float32)  # 形状: [n, 5]
    
    # 创建PyTorch数据集
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return dataset, scaler

if __name__ == '__main__':
    # 使用kagglehub下载数据集
    import kagglehub
    data_path = kagglehub.dataset_download("smid80/weatherww2")
    print("数据集文件路径:", data_path)
    
    # 设置TensorBoard日志路径
    run_file_path = "./runs/weather_rnn"
    writer = SummaryWriter(log_dir=run_file_path)
    
    # 设置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载和准备数据
    dataset, scaler = load_weather_data(f"{data_path}/Summary of Weather.csv")
    
    # 分割训练集和测试集
    train_size = int(0.8 * len(dataset))  # 80%用于训练
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset)-train_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    
    # 初始化模型、损失函数和优化器
    model = WeatherRNN().to(device)
    
    # 对于回归任务，使用均方误差损失而不是交叉熵损失
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10  # 训练轮数
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到指定设备
            
            # 前向传播
            optimizer.zero_grad()  # 清除梯度
            outputs = model(inputs)  # 模型预测
            loss = criterion(outputs, targets)  # 计算损失
            
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            
            total_loss += loss.item()
            
            # 每10个批次打印一次训练状态
            if i % 10 == 0:
                print(f'轮次 [{epoch+1}/{num_epochs}], 步骤 [{i}/{len(train_loader)}], 损失: {loss.item():.4f}')
                writer.add_scalar('训练损失', loss.item(), epoch * len(train_loader) + i)
        
        # 计算平均训练损失
        avg_loss = total_loss / len(train_loader)
        print(f'轮次 [{epoch+1}/{num_epochs}], 平均训练损失: {avg_loss:.4f}')
        
        # 评估阶段
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度
            test_loss = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # 模型预测
                loss = criterion(outputs, targets)  # 计算测试损失
                test_loss += loss.item()
            
            # 计算平均测试损失
            avg_test_loss = test_loss / len(test_loader)
            print(f'轮次 [{epoch+1}/{num_epochs}], 测试损失: {avg_test_loss:.4f}')
            writer.add_scalar('测试损失', avg_test_loss, epoch)
    
    # 保存模型
    torch.save(model.state_dict(), 'weather_rnn_model.pth')
    print("模型已保存.")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 使用训练好的模型进行预测的示例
    print("\n使用训练好的模型进行预测:")
    model.eval()
    with torch.no_grad():
        # 从测试数据中获取样本
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 进行预测
        predictions = model(inputs)
        
        # 将预测结果转换回原始比例
        predictions_np = predictions.cpu().numpy()
        predictions_reshaped = predictions_np.reshape(-1, 5)  # 5是预测天数
        
        # 显示第一个数据点的样本预测值与实际值
        print("\n样本预测 (归一化值):")
        print(f"预测未来5天: {predictions[0].cpu().numpy()}")
        print(f"实际未来5天: {targets[0].cpu().numpy()}")
        
        # 注意：要获取真实温度，需要使用scaler进行反向转换
        # 例如：scaler.inverse_transform(predictions_np.reshape(-1, 1)).reshape(predictions_reshaped.shape)
