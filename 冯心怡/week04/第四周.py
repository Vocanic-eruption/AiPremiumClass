# 导入必要包
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_olivetti_faces

# 定义超参数
LR = 1e-3
epochs = 20
BATCH_SIZE = 32

# 数据集加载
olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)

# 训练集
train_data = olivetti_faces.data[:300]
train_data = torch.tensor(train_data, dtype=torch.float32)
train_label = olivetti_faces.target[:300]
train_label = torch.tensor(train_label, dtype=torch.int64)

# 测试集
test_data = olivetti_faces.data[300:]
test_data = torch.tensor(test_data, dtype=torch.float32)
test_label = olivetti_faces.target[300:]
test_label = torch.tensor(test_label, dtype=torch.int64)

# 将训练数据和标签包装成 TensorDataset
train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)

# 创建 DataLoader
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义优化后的模型
class TorchNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 256)  # 输入维度 4096
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, 40)  # 输出维度 40（40 个类别）
        self.drop = nn.Dropout(p=0.5)  # 增加 Dropout 比例
        self.act = nn.ReLU()

    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.act(out)
        out = self.drop(out)
        final = self.linear4(out)
        return final

# 初始化模型
model = TorchNN()

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

# 定义不同的优化器
optimizers = {
    "SGD": optim.SGD(model.parameters(), lr=LR),
    "Adam": optim.Adam(model.parameters(), lr=LR),
    "RMSprop": optim.RMSprop(model.parameters(), lr=LR),
    "AdamW": optim.AdamW(model.parameters(), lr=LR)
}

# 训练和测试函数
def train_and_test(optimizer, name):
    print(f"\n使用优化器: {name}")
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        for data, target in train_dl:
            # 前向运算
            output = model(data)
            # 计算损失
            loss = loss_fn(output, target)
            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

        print(f'Epoch:{epoch + 1}/{epochs}, Loss: {loss.item()}')

    # 测试模型
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_dl:
            output = model(data)
            _, predicted = torch.max(output, 1)  # 获取预测结果
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'测试集准确率: {100 * correct / total:.2f}%')

# 使用不同的优化器训练和测试模型
for name, optimizer in optimizers.items():
    train_and_test(optimizer, name)
