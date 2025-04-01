import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
def load_data(batch_size=128):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
    ])
    
    # 加载KMNIST数据集
    train_dataset = torchvision.datasets.KMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.KMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader

# 定义简单的全连接神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, hidden_size=128, num_hidden_layers=1):
        super(SimpleNN, self).__init__()
        
        layers = []
        input_size = 28 * 28  # KMNIST图像大小为28x28
        
        # 添加第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # 添加额外的隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # 添加输出层
        layers.append(nn.Linear(hidden_size, 10))  # KMNIST有10个类别
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        return self.model(x)

# 定义卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self, num_filters1=32, num_filters2=64):
        super(ConvNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, num_filters1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 计算全连接层的输入大小
        # 经过两次池化后，图像尺寸变为原来的1/4
        fc_input_size = num_filters2 * (28 // 4) * (28 // 4)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

# 训练模型
def train_model(model, train_loader, test_loader, learning_rate=0.001, num_epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        
        # 记录结果
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Time: {epoch_time:.2f}s, '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Acc: {test_accuracy:.2f}%')
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies

# 可视化训练过程
def plot_learning_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

# 可视化混淆矩阵
def plot_confusion_matrix(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# 可视化一些样本
def visualize_samples(test_loader, num_samples=10):
    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 显示图像
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

# 实验不同的模型结构
def experiment_model_structures(batch_size=128, learning_rate=0.001, num_epochs=10):
    train_loader, test_loader = load_data(batch_size)
    
    # 可视化部分样本
    visualize_samples(test_loader)
    
    # 实验1：基本的全连接网络
    print("\n实验1：基本的全连接网络")
    model1 = SimpleNN(hidden_size=128, num_hidden_layers=1)
    model1, train_losses1, test_losses1, train_accs1, test_accs1 = train_model(
        model1, train_loader, test_loader, learning_rate, num_epochs
    )
    
    # 实验2：增加更多隐藏层的全连接网络
    print("\n实验2：增加更多隐藏层的全连接网络")
    model2 = SimpleNN(hidden_size=128, num_hidden_layers=3)
    model2, train_losses2, test_losses2, train_accs2, test_accs2 = train_model(
        model2, train_loader, test_loader, learning_rate, num_epochs
    )
    
    # 实验3：更大的神经网络
    print("\n实验3：更大的神经网络")
    model3 = SimpleNN(hidden_size=256, num_hidden_layers=2)
    model3, train_losses3, test_losses3, train_accs3, test_accs3 = train_model(
        model3, train_loader, test_loader, learning_rate, num_epochs
    )
    
    # 实验4：卷积神经网络
    print("\n实验4：卷积神经网络")
    model4 = ConvNet(num_filters1=32, num_filters2=64)
    model4, train_losses4, test_losses4, train_accs4, test_accs4 = train_model(
        model4, train_loader, test_loader, learning_rate, num_epochs
    )
    
    # 比较不同模型的性能
    plt.figure(figsize=(12, 5))
    
    # 绘制测试准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(test_accs1, label='Basic NN')
    plt.plot(test_accs2, label='Deeper NN')
    plt.plot(test_accs3, label='Wider NN')
    plt.plot(test_accs4, label='CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy Comparison')
    
    # 绘制测试损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_losses1, label='Basic NN')
    plt.plot(test_losses2, label='Deeper NN')
    plt.plot(test_losses3, label='Wider NN')
    plt.plot(test_losses4, label='CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Test Loss Comparison')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # 为表现最好的模型绘制混淆矩阵
    best_model = model4
    plot_confusion_matrix(best_model, test_loader)
    
    return best_model

# 超参数调优：学习率和批次大小
def tune_hyperparameters():
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [32, 128, 256]
    
    results = {}
    
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n尝试学习率={lr}, 批次大小={bs}")
            
            train_loader, test_loader = load_data(batch_size=bs)
            model = ConvNet()  # 使用CNN作为基础模型
            
            # 只训练5个epoch
            _, _, _, _, test_accs = train_model(
                model, train_loader, test_loader, learning_rate=lr, num_epochs=5
            )
            
            # 保存最终测试准确率
            results[(lr, bs)] = test_accs[-1]
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    # 创建一个表格布局
    cell_text = []
    for bs in batch_sizes:
        row = []
        for lr in learning_rates:
            row.append(f"{results[(lr, bs)]:.2f}%")
        cell_text.append(row)
    
    plt.table(cellText=cell_text,
              rowLabels=[f'BS={bs}' for bs in batch_sizes],
              colLabels=[f'LR={lr}' for lr in learning_rates],
              loc='center')
    
    plt.axis('off')
    plt.title('Test Accuracy (%) for Different Learning Rates and Batch Sizes')
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning.png')
    plt.show()
    
    # 找出最佳超参数
    best_params = max(results, key=results.get)
    best_lr, best_bs = best_params
    best_acc = results[best_params]
    
    print(f"\n最佳超参数: 学习率={best_lr}, 批次大小={best_bs}, 准确率={best_acc:.2f}%")
    
    return best_lr, best_bs

# 主函数
def main():
    print("开始实验不同的模型结构...")
    best_model = experiment_model_structures(num_epochs=10)
    
    print("\n开始超参数调优...")
    best_lr, best_bs = tune_hyperparameters()
    
    print("\n使用最佳超参数训练最终模型...")
    train_loader, test_loader = load_data(batch_size=best_bs)
    
    final_model = ConvNet()
    final_model, train_losses, test_losses, train_accs, test_accs = train_model(
        final_model, train_loader, test_loader, learning_rate=best_lr, num_epochs=15
    )
    
    # 可视化最终模型的学习曲线
    plot_learning_curves(train_losses, test_losses, train_accs, test_accs)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(final_model, test_loader)
    
    # 保存最终模型
    torch.save(final_model.state_dict(), 'kmnist_final_model.pth')
    print("最终模型已保存为 'kmnist_final_model.pth'")

if __name__ == "__main__":
    main() 
