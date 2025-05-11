import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self,):
        super().__init__()
        # self.rnn = nn.RNN(
        self.rnn = nn.LSTM(
            input_size=64,   # x的特征维度
            hidden_size=50,  # 隐藏层神经元数量 w_ht[50,64], w_hh[50,50]
            bias=True,        # 偏置[50]
            # num_layers=1,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(50, 40)  # 输出层 

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out


if __name__ == '__main__':

    run_file_path = r"d:\vsCodeProj\AiPremiumClass\王健钢\week6_循环神经网络\run"
    writer = SummaryWriter(log_dir=run_file_path)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # 加载数据集
    # 从sklearn中获取olivetti_faces数据集
    olivetti_faces = fetch_olivetti_faces(data_home='D:\\datasets\\face_data', shuffle=True) 

    # 加载数据集
    X, y = olivetti_faces.images, olivetti_faces.target

    # 适配格式 [batch, time_steps, features]
    X = X.reshape(-1, 64, 64).astype('float32')
    y = torch.tensor(y, dtype=torch.long)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test), y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # 实例化模型
    model = RNN_Classifier()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], 64, 64)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
        # 评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer.add_scalar('test accuracy', accuracy, epoch)

    # 记录模型结构到 TensorBoard
    sample_input = torch.zeros(1, 64, 64).to(device)  # 1 张人脸数据
    writer.add_graph(model, sample_input)

    writer.close()

