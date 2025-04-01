import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载Olivetti人脸数据集
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data
y = faces.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建数据加载器
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 定义模型类
class BaseModel(nn.Module):
    def __init__(self, input_dim=4096):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class NormalizedModel(nn.Module):
    def __init__(self, input_dim=4096):
        super(NormalizedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class RegularizedModel(nn.Module):
    def __init__(self, input_dim=4096):
        super(RegularizedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 40)
        # 注意：PyTorch中正则化通常在优化器中设置，而不是在模型定义中

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


class NormRegModel(nn.Module):
    def __init__(self, input_dim=4096):
        super(NormRegModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 40)
        # 正则化在优化器中设置

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# 4. 训练函数
def train_model(model, optimizer, criterion, train_loader, test_loader, epochs=50):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(test_loader))

        # 每10个epoch输出一次结果
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {correct / total:.4f}')

    # 计算最终准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = correct / total

    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_accuracy': final_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# 5. 实验：不同的模型结构和优化器
models = [
    (BaseModel, 'Base Model'),
    (NormalizedModel, 'Normalized Model'),
    (RegularizedModel, 'Regularized Model'),
    (NormRegModel, 'Normalized+Regularized Model')
]

optimizers = [
    ('SGD', lambda model: optim.SGD(model.parameters(), lr=0.01)),
    ('Adam', lambda model: optim.Adam(model.parameters(), lr=0.001)),
    ('RMSprop', lambda model: optim.RMSprop(model.parameters(), lr=0.001))
]

criterion = nn.CrossEntropyLoss()
epochs = 50

# 6. 运行实验
results = []

print("=== 训练所有模型 ===")
for model_class, model_name in models:
    for opt_name, opt_func in optimizers:
        # 对于正则化模型，添加L1和L2正则化
        if model_name in ['Regularized Model', 'Normalized+Regularized Model']:
            model = model_class()
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # L2正则化
            elif opt_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            else:  # RMSprop
                optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            model = model_class()
            optimizer = opt_func(model)

        print(f"\n训练 {model_name} with {opt_name}:")
        result = train_model(model, optimizer, criterion, train_loader, test_loader, epochs)

        results.append({
            'model_name': model_name,
            'optimizer': opt_name,
            'final_train_loss': result['final_train_loss'],
            'final_val_loss': result['final_val_loss'],
            'final_accuracy': result['final_accuracy']
        })

        print(
            f"{model_name} with {opt_name} - Val Loss: {result['final_val_loss']:.4f}, Accuracy: {result['final_accuracy']:.4f}")

# 7. 结果分析
print("\n=== 结果分析 ===")

# 比较不同模型结构的表现
model_avg_loss = {}
for result in results:
    if result['model_name'] not in model_avg_loss:
        model_avg_loss[result['model_name']] = {'loss': [], 'acc': []}
    model_avg_loss[result['model_name']]['loss'].append(result['final_val_loss'])
    model_avg_loss[result['model_name']]['acc'].append(result['final_accuracy'])

print("\n模型结构比较：")
for model_name, stats in model_avg_loss.items():
    avg_loss = np.mean(stats['loss'])
    avg_acc = np.mean(stats['acc'])
    print(f"{model_name} - 平均验证损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.4f}")

# 比较不同优化器的表现
optimizer_avg_loss = {}
for result in results:
    if result['optimizer'] not in optimizer_avg_loss:
        optimizer_avg_loss[result['optimizer']] = {'loss': [], 'acc': []}
    optimizer_avg_loss[result['optimizer']]['loss'].append(result['final_val_loss'])
    optimizer_avg_loss[result['optimizer']]['acc'].append(result['final_accuracy'])

print("\n优化器比较：")
for optimizer_name, stats in optimizer_avg_loss.items():
    avg_loss = np.mean(stats['loss'])
    avg_acc = np.mean(stats['acc'])
    print(f"{optimizer_name} - 平均验证损失: {avg_loss:.4f}, 平均准确率: {avg_acc:.4f}")

# 找出最佳模型
best_model = min(results, key=lambda x: x['final_val_loss'])
print(
    f"\n最佳模型: {best_model['model_name']} with {best_model['optimizer']} - 验证损失: {best_model['final_val_loss']:.4f}, 准确率: {best_model['final_accuracy']:.4f}")