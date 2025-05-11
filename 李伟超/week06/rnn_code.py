import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,   # 输入特征维度（调整为 64 以适配 Olivetti Faces）
            hidden_size=128,  # 隐藏层神经元数量
            num_layers=3,     # 降低层数避免过拟合
            batch_first=True  # 批次维度在第一维
        )
        self.fc = nn.Linear(128, 40)  # Olivetti Faces 有 40 类

    def forward(self, x):
        # 输入 x 形状: [batch, time_steps, features]
        outputs, _ = self.rnn(x)
        out = self.fc(outputs[:, -1, :])  # 取最后时间步的输出
        return out


if __name__ == '__main__':

    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    faces = fetch_olivetti_faces(shuffle=True)
    X, y = faces.images, faces.target

    # 归一化 & 调整形状为 RNN 适配格式 [batch, time_steps, features]
    X = X.reshape(-1, 64, 64).astype('float32')  # 64×64 作为时间步 & 特征
    y = torch.tensor(y, dtype=torch.long)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test), y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型
    model = RNN_Classifier().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], 64, 64)  # 调整为 (batch, time_steps, features)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('average training loss', avg_loss, epoch)

        # 评估模型
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], 64, 64)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('test accuracy', accuracy, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # 记录模型结构到 TensorBoard
    sample_input = torch.zeros(1, 64, 64).to(device)  # 1 张人脸数据
    writer.add_graph(model, sample_input)

    # 保存模型
    # torch.save(model, 'rnn_model.pth')
    torch.save(model.state_dict(), 'rnn_model_params.pth')

    writer.close()

    # 加载模型
    # model = torch.load('rnn_model.pth')

    # 加载模型参数
    model = RNN_Classifier()
    model.load_state_dict(torch.load('rnn_model_params.pth'))
    model.to(device)
