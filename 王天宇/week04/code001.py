import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 引入数据
olivetti_faces = fetch_olivetti_faces(data_home='./', shuffle=True)

# 数据预处理
X = olivetti_faces.data
y = olivetti_faces.target

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader
batch_size = 32  # 你可以根据需要调整batch_size
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义神经网络模型
class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.linear1 = nn.Linear(4096, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 40)  # 输出层，40个类别
        self.drop = nn.Dropout(p=0.3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn1(self.linear1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.linear2(x)))
        x = self.drop(x)
        x = self.linear3(x)
        return x

# 初始化模型
model1 = FaceRecognitionModel()

# 定义损失函数和优化器
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model1.train()
    running_loss = 0.0
    for data, target in train_dl:
        # 前向计算
        outputs = model1(data) 
        loss = loss_func(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"epoch: {epoch+1}/{epochs}, loss = {running_loss/len(train_dl):.4f}")