from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from nn_module import SimplNN

# 1. 搭建的神经网络，使用olivettiface数据集进行训练。
# 2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
# 3. 尝试不同optimizer对模型进行训练，观察对比loss结果。
# 4. 注册kaggle并尝试激活Accelerator，使用GPU加速模型训练。




if __name__ == '__main__':
    # 加载数据集

    faces = fetch_olivetti_faces(data_home='./党金虎/week04/scikit_learn_data')
    X = faces.data # (400, 4096) 400张图片，    每张图片64*64=4096个像素点
    y = faces.target # (400,) 400个标签
    print(X.shape, y.shape)

    # 转 pytorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 划分数据集 训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 80%训练集，20%测试集 

    train_dataset = TensorDataset(X_train, y_train) # 将训练数据集转换为张量
    test_dataset = TensorDataset(X_test, y_test) # 将测试数据集转换为张量

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # 将训练数据集转换为加载器 32个样本一组 shuffle打乱顺序
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False) # 将测试数据集转换为加载器 32个样本一组 不打乱顺序

    #训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 确认是否有GPU
    model = SimplNN().to(device) # 将模型移动到GPU上
    criterion = nn.CrossEntropyLoss() # 损失函数

    # 定义不同优化器
    optimizers = {
        'SGD': torch.optim.SGD(model.parameters(), lr=0.001),
        #'Adam': torch.optim.Adam(model.parameters(), lr=0.01),
         'Adam': torch.optim.Adam(model.parameters(),lr=0.01),
    }

    # 训练和测试
    results = {}

    for optimizer_name, optimizer in optimizers.items():
        print(f'训练 Optimizer: {optimizer_name}')
        train_losses = []
        test_losses = []

        for epoch in range(80): # 训练20轮
            train_loss = model.model_train( train_loader, criterion, optimizer, device)
            test_loss = model.model_test( test_loader, criterion, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f'Epoch: [{epoch+1}/ 20], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        results[optimizer_name] = {'train_losses': train_losses, 'test_losses': test_losses}

    # 可视化
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    for optimizer_name, result in results.items():
        plt.plot(result["train_losses"], label=f"{optimizer_name} Train Loss")
        plt.plot(result["test_losses"], label=f"{optimizer_name} Test Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss with Different Optimizers")
    plt.legend()
    plt.show()