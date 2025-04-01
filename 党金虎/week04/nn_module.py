import torch
import torch.nn as nn


# 定义神经网络模型
class SimplNN(nn.Module):
    def __init__(self):
        super(SimplNN, self).__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.bn1= nn.BatchNorm1d(512) # 批量归一化层
        self.fc2 = nn.Linear(512,256)
        self.bn2= nn.BatchNorm1d(256) # 批量归一化层
        self.fc3 = nn.Linear(256, 40) # 输出层
        self.dropout = nn.Dropout(p=0.5) # Dropout 正则化


    # 定义前向传播
    def forward(self, x):
        x = self.fc1(x) # 第一层全连接
        x = self.bn1(x) # 批量归一化
        x = torch.relu(x) # relu激活函数
        x = self.dropout(x) # Dropout 正则化
        x = self.fc2(x) # 第二层全连接
        x = self.bn2(x) # 批量归一化
        x = torch.relu(x) # relu激活函数
        x = self.dropout(x) # Dropout 正则化
        x = self.fc3(x) # 输出层
        return x
    
    # 定义训练函数,参数分别为模型，训练数据，损失函数，优化器，设备
    def model_train( self,train_loader, criterion, optimizer, device): 
        self.train() # 设置为训练模式
        running_loss = 0.0 # 初始化损失
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # 梯度清零

            outputs = self(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            running_loss += loss.item() # 累加损失
        return running_loss / len(train_loader) # 返回平均损失

    # 定义测试函数，参数分别为模型，测试数据，损失函数，设备
    def model_test(self, test_loader, criterion, device):
        self.eval() 
        '''eval的作用是将模型设置为评估模式。在评估模式下,模型的行为会有所不同，特别是在涉及到某些层（如 Dropout 和 BatchNorm）时。具体来说：
    Dropout 层：在训练模式下，Dropout 层会随机丢弃一部分神经元以防止过拟合；而在评估模式下，Dropout 层会关闭，不再丢弃神经元。
    BatchNorm 层：在训练模式下，BatchNorm 层会使用当前批次的均值和方差进行归一化；而在评估模式下，BatchNorm 层会使用整个训练集的均值和方差进行归一化。
    通过调用 model.eval()，可以确保在评估或测试模型时，模型的行为与训练时一致，从而得到更稳定和可靠的评估结果。'''
        
        running_loss = 0.0 # 初始化损失
        with torch.no_grad(): # 关闭梯度计算
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs) # 前向传播
                loss = criterion(outputs, labels) # 计算损失
                running_loss += loss.item() #  累加损失
        return running_loss / len(test_loader) # 返回平均损失


