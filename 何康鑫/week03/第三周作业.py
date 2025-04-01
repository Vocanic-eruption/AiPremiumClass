import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 数据预处理
transform = transforms.Compose([    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载KMNIST数据集
train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义模型结构（3层全连接）
class KMNISTClassifier(nn.Module):
    def __init__(self):
        super(KMNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(256, 128)    # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(128, 10)     # 隐藏层2到输出层
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = KMNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

class EnhancedModel(nn.Module):
    def __init__(self):
        super(EnhancedModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 增加神经元数量
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)   # 新增隐藏层
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class BNModel(nn.Module):
    def __init__(self):
        super(BNModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)  # 批归一化层
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        return self.fc3(x)

# 使用StepLR调度器（每3个epoch衰减0.1倍）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
