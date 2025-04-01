import torch
from sklearn.datasets import fetch_olivetti_faces
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.fc1 = nn.Linear(4096, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 40)

    # 更新参数
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x



# 定义超参数
LR = 1e-2
epochs = 50
BATCH_SIZE = 64

# 数据集加载
olivetti_faces = fetch_olivetti_faces(data_home='./scikit_learn_data', shuffle=True)
# print(X.shape, y.shape)
# face = olivetti_faces.images[5]
# plt.imshow(face, cmap='gray')
# plt.show()

images = torch.tensor(olivetti_faces.data)
target = torch.tensor(olivetti_faces.target)
dataset = [(img, lbl) for img, lbl in zip(images, target)]

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 创建模型实例
model = FaceModel()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数

# 优化器
optimizer = optim.SGD(model.parameters(), lr = LR)

# 训练模式
model.train()

# 评估模式
# model.eval()

# 训练模型
for epoch in range(epochs):
    # 提取训练数据
    for img, lbl in dataloader:
        # 向前运算
        oupput = model(img.reshape(-1, 4096))
        # 计算损失
        loss = loss_fn(oupput, lbl)
        # 反向传播
        optimizer.zero_grad() # 所有参数梯度清零
        loss.backward()  # 计算梯度
        optimizer.step() # 更新参数


    print(f'Epoch: {epoch}, Loss: {loss.item()}')



