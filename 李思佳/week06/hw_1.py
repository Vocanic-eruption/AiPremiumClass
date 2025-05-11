import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self,type):
        super().__init__()
        if type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=64,   # x的特征维度
                hidden_size=128,  # 隐藏层神经元数量 w_ht[50,4096], w_hh[50,50]
                bias=True,        # 偏置[50]
                num_layers=5,     # 隐藏层层数
                batch_first=True  # 批次是输入第一个维度
            )
        if type == "GRU":
                self.rnn = nn.GRU(
                input_size=64,   # x的特征维度
                hidden_size=128,  # 隐藏层神经元数量 w_ht[50,28], w_hh[50,50]
                bias=True,        # 偏置[50]
                num_layers=5,     # 隐藏层层数
                batch_first=True  # 批次是输入第一个维度
            )
        self.fc = nn.Linear(128, 40)  # 输出层 

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out


if __name__ == '__main__':

    # writer = SummaryWriter(log_dir='李思佳/week06/logs')
    writer_lstm = SummaryWriter(log_dir='李思佳/week06/logs/LSTM')
    writer_gru = SummaryWriter(log_dir='李思佳/week06/logs/GRU')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    X, y = fetch_olivetti_faces(data_home='./李思佳/week06/face_data',return_X_y=True, shuffle=True)
    X = X.reshape(-1, 64, 64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    # train_data_set = [(img,lbl) for img,lbl in zip(X_train, y_train)]
    # test_data_set = [(img,lbl) for img,lbl in zip(X_test, y_test)]
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型
    model = RNN_Classifier(type="LSTM")
    model.to(device)

    model2 = RNN_Classifier(type="GRU")
    model2.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)


    # 训练模型
    num_epochs = 50
    print("LSTM模型:")
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()

            # 记录训练损失
            writer_lstm.add_scalar('lstm Training Loss', loss.item(), epoch * len(train_loader) + i)

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                # writer_lstm.add_scalar('lstm training loss', loss.item(), epoch * len(train_loader) + i)
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
            writer_lstm.add_scalar('LSTM Test Accuracy', accuracy, epoch)
    


    # 训练模型
    num_epochs = 50
    print("GRU模型:")
    for epoch in range(num_epochs):
        model2.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model2(images.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model2.parameters(),max_norm=1.0)
            optimizer.step()
            writer_gru.add_scalar('gru Training Loss', loss.item(), epoch * len(train_loader) + i)

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                # writer_gru.add_scalar('gru training loss', loss.item(), epoch * len(train_loader) + i)
        # 评估模型
        model2.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model2(images.squeeze())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer_gru.add_scalar('gru test accuracy', accuracy, epoch)

    writer_lstm.close() 
    writer_gru.close()  # 关闭 SummaryWriter


