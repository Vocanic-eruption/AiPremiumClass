import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Set global font to Times New Roman for plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300  # Higher DPI for better quality

# 定义超参数
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 32

# 加载olivettiface数据集
print("加载olivettiface数据集...")
olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
X = olivetti_faces.data  # 形状为(400, 4096)
y = olivetti_faces.target  # 形状为(400,)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义基础神经网络模型
class OlivettiFaceNet(nn.Module):
    def __init__(self, use_bn=False, dropout_rate=0.0):
        super().__init__()
        self.use_bn = use_bn
        
        # 第一层
        self.fc1 = nn.Linear(4096, 1024)
        self.bn1 = nn.BatchNorm1d(1024) if use_bn else None
        
        # 第二层
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512) if use_bn else None
        
        # 第三层
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256) if use_bn else None
        
        # 输出层
        self.fc4 = nn.Linear(256, 40)  # 40个人的脸
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        # 第一层
        x = self.fc1(x)
        if self.use_bn and self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.fc2(x)
        if self.use_bn and self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.fc3(x)
        if self.use_bn and self.bn3 is not None:
            x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc4(x)
        return x

# 训练函数
def train_model(model, train_loader, optimizer, criterion, device, epochs=EPOCHS):
    model.train()
    model.to(device)
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    
    return losses

# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 实验1: 归一化实验
def experiment_normalization():
    print("\nExperiment 1: Normalization Experiment")
    
    # 基础模型
    base_model = OlivettiFaceNet(use_bn=False, dropout_rate=0.0)
    base_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=LR)
    
    print("Training Base Model...")
    base_losses = train_model(base_model, train_loader, optimizer, criterion, device)
    base_accuracy = evaluate_model(base_model, test_loader, device)
    
    # 使用BatchNorm的模型
    bn_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.0)
    bn_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bn_model.parameters(), lr=LR)
    
    print("Training BatchNorm Model...")
    bn_losses = train_model(bn_model, train_loader, optimizer, criterion, device)
    bn_accuracy = evaluate_model(bn_model, test_loader, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(base_losses, label='Base Model', color='#1f77b4', linewidth=2)
    plt.plot(bn_losses, label='BatchNorm Model', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Normalization Comparison Experiment')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('normalization_experiment.png')
    plt.show()
    
    return base_accuracy, bn_accuracy

# 实验2: 使用不同的优化器
def experiment_optimizers():
    print("\nExperiment 2: Optimizer Comparison")
    
    # 使用Adam优化器
    adam_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.3)
    adam_model.to(device)
    criterion = nn.CrossEntropyLoss()
    adam_optimizer = optim.Adam(adam_model.parameters(), lr=LR)
    
    print("Training Model with Adam Optimizer...")
    adam_losses = train_model(adam_model, train_loader, adam_optimizer, criterion, device)
    adam_accuracy = evaluate_model(adam_model, test_loader, device)
    
    # 使用SGD优化器
    sgd_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.3)
    sgd_model.to(device)
    criterion = nn.CrossEntropyLoss()
    sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=LR, momentum=0.9)
    
    print("Training Model with SGD Optimizer...")
    sgd_losses = train_model(sgd_model, train_loader, sgd_optimizer, criterion, device)
    sgd_accuracy = evaluate_model(sgd_model, test_loader, device)
    
    # 使用RMSprop优化器
    rmsprop_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.3)
    rmsprop_model.to(device)
    criterion = nn.CrossEntropyLoss()
    rmsprop_optimizer = optim.RMSprop(rmsprop_model.parameters(), lr=LR)
    
    print("Training Model with RMSprop Optimizer...")
    rmsprop_losses = train_model(rmsprop_model, train_loader, rmsprop_optimizer, criterion, device)
    rmsprop_accuracy = evaluate_model(rmsprop_model, test_loader, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(adam_losses, label='Adam', color='#1f77b4', linewidth=2)
    plt.plot(sgd_losses, label='SGD', color='#ff7f0e', linewidth=2)
    plt.plot(rmsprop_losses, label='RMSprop', color='#2ca02c', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison Experiment')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimizer_experiment.png')
    plt.show()
    
    return adam_accuracy, sgd_accuracy, rmsprop_accuracy

# 实验3: 正则化实验 (Dropout)
def experiment_regularization():
    print("\nExperiment 3: Regularization Experiment (Dropout)")
    
    # 不使用Dropout
    no_dropout_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.0)
    no_dropout_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(no_dropout_model.parameters(), lr=LR)
    
    print("Training Model without Dropout...")
    no_dropout_losses = train_model(no_dropout_model, train_loader, optimizer, criterion, device)
    no_dropout_accuracy = evaluate_model(no_dropout_model, test_loader, device)
    
    # 使用Dropout
    dropout_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.3)
    dropout_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dropout_model.parameters(), lr=LR)
    
    print("Training Model with Dropout...")
    dropout_losses = train_model(dropout_model, train_loader, optimizer, criterion, device)
    dropout_accuracy = evaluate_model(dropout_model, test_loader, device)
    
    # 使用L2正则化 (权重衰减)
    l2_model = OlivettiFaceNet(use_bn=True, dropout_rate=0.0)
    l2_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(l2_model.parameters(), lr=LR, weight_decay=1e-4)  # 添加L2正则化
    
    print("Training Model with L2 Regularization...")
    l2_losses = train_model(l2_model, train_loader, optimizer, criterion, device)
    l2_accuracy = evaluate_model(l2_model, test_loader, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(no_dropout_losses, label='No Regularization', color='#1f77b4', linewidth=2)
    plt.plot(dropout_losses, label='Dropout', color='#ff7f0e', linewidth=2)
    plt.plot(l2_losses, label='L2 Regularization', color='#2ca02c', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regularization Comparison Experiment')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('regularization_experiment.png')
    plt.show()
    
    return no_dropout_accuracy, dropout_accuracy, l2_accuracy

# 主函数
def main():
    print("Starting Neural Network Training Experiments on Olivetti Face Dataset")
    
    # 实验1: 归一化对比
    base_acc, bn_acc = experiment_normalization()
    
    # 实验2: 不同优化器对比
    adam_acc, sgd_acc, rmsprop_acc = experiment_optimizers()
    
    # 实验3: 正则化实验
    no_reg_acc, dropout_acc, l2_acc = experiment_regularization()
    
    # 打印所有实验结果
    print("\nSummary of All Experiment Results:")
    print(f"Normalization Experiment - Base Model: {base_acc:.2f}%, BatchNorm Model: {bn_acc:.2f}%")
    print(f"Optimizer Experiment - Adam: {adam_acc:.2f}%, SGD: {sgd_acc:.2f}%, RMSprop: {rmsprop_acc:.2f}%")
    print(f"Regularization Experiment - No Regularization: {no_reg_acc:.2f}%, Dropout: {dropout_acc:.2f}%, L2 Regularization: {l2_acc:.2f}%")

if __name__ == "__main__":
    main()
