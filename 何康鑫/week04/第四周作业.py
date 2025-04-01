import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 加载数据集
olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
images = olivetti_faces.images.reshape((-1, 64 * 64))  # 转换为2D数据
labels = olivetti_faces.target

# 数据标准化处理
mean = images.mean(axis=0)
std = images.std(axis=0)
images = (images - mean) / std

# 创建数据加载器
dataset = TensorDataset(torch.tensor(images, dtype=torch.float32),
                        torch.tensor(labels, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型（修正输入维度为64x64=4096）
class OlivettiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 1024)  # 输入层修正为4096
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 40)     # 输出层为40类
        self.drop = nn.Dropout(p=0.3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OlivettiNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 训练循环
loss_history = []
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_img, batch_lbl in dataloader:
        batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_img.view(-1, 4096))  # 修正reshape维度
        loss = loss_fn(outputs, batch_lbl)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

# 绘制loss曲线
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
class OlivettiNetNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 40)
        self.drop = nn.Dropout(p=0.3)  # 仅保留Dropout
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

# 训练并记录loss
model_no_bn = OlivettiNetNoBN().to(device)
optimizer_no_bn = optim.AdamW(model_no_bn.parameters(), lr=1e-3, weight_decay=1e-4)
loss_history_no_bn = []

for epoch in range(epochs):
    model_no_bn.train()
    running_loss = 0.0
    for batch_img, batch_lbl in dataloader:
        batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
        
        optimizer_no_bn.zero_grad()
        outputs = model_no_bn(batch_img.view(-1, 4096))
        loss = loss_fn(outputs, batch_lbl)
        loss.backward()
        optimizer_no_bn.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    loss_history_no_bn.append(epoch_loss)
    print(f"No BN Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

# 对比结果
plt.plot(loss_history, label='Baseline (BN+Dropout)')
plt.plot(loss_history_no_bn, label='No BN')
plt.legend()
plt.title('Comparison of BN vs No BN')
plt.show()
optimizers = {
    'AdamW': optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
    'SGD': optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
    'RMSprop': optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99),
    'Adam': optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
}

loss_histories = {}

for name, optimizer in optimizers.items():
    model = OlivettiNet().to(device)
    loss_hist = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_img, batch_lbl in dataloader:
            batch_img, batch_lbl = batch_img.to(device), batch_lbl.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_img.view(-1, 4096))
            loss = loss_fn(outputs, batch_lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        loss_hist.append(epoch_loss)
        print(f"{name} Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")
    
    loss_histories[name] = loss_hist

# 绘制对比图
plt.figure(figsize=(12,6))
for name, hist in loss_histories.items():
    plt.plot(hist, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.show()
