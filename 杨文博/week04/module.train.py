from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(4096, 8192)
        self.bn1 = nn.BatchNorm1d(8192)

        self.linear2 = nn.Linear(8192, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.linear4 = nn.Linear(1024, 40)
        self.bn4 = nn.BatchNorm1d(40)

        self.activation = Swish()
        self.dropout = nn.Dropout()
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear4(out)
        return out

livetti_faces = fetch_olivetti_faces(data_home="./face_data", shuffle=True)
face = livetti_faces.images[0]
plt.imshow(face,cmap='gray')
plt.show()

device = torch.device("cpu")

images = torch.tensor(livetti_faces.data)
targets = torch.tensor(livetti_faces.target)
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

model = FaceModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(40):
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        result = model(images)
        loss = criterion(result, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(i, loss.item())


correct = 0  # 记录正确预测的数量
total = 0    # 记录总样本数量

model.eval()  # 切换到评估模式（关闭 Dropout、BatchNorm 等）
with torch.no_grad():  # 关闭梯度计算，加快计算速度，减少显存占用
    for images, labels in test_dataloader:  # 遍历测试集
        images, labels = images.to(device), labels.to(device)  # 迁移到 GPU/CPU

        outputs = model(images)  # 获取模型预测结果
        _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别索引
        correct += (predicted == labels).sum().item()  # 统计预测正确的样本数
        total += labels.size(0)  # 统计总样本数

accuracy = correct / total  # 计算准确率
print(f"Accuracy: {accuracy * 100:.2f}%")

