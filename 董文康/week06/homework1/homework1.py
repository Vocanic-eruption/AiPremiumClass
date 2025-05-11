from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self, model_name='RNN'):
        super().__init__()
        if model_name == 'RNN':
            self.rnn = nn.RNN(
                input_size=64,   # x的特征维度
                hidden_size=256,  # 隐藏层神经元数量 
                bias=True,        # 偏置
                num_layers=5,     # 隐藏层层数
                batch_first=True  # 批次是输入第一个维度
            )
        elif  model_name == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=64,   # x的特征维度
                hidden_size=256,  # 隐藏层神经元数量 
                bias=True,        # 偏置
                num_layers=5,     # 隐藏层层数
                batch_first=True  # 批次是输入第一个维度
            )
        elif model_name == 'GRU':
            self.rnn = nn.GRU(
                input_size=64,   # x的特征维度
                hidden_size=256,  # 隐藏层神经元数量 
                bias=True,        # 偏置
                num_layers=5,     # 隐藏层层数
                batch_first=True  # 批次是输入第一个维度
            )
        self.fc = nn.Linear(256, 40)  # 输出层 

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out

olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)

images = torch.tensor(olivetti_faces.images)
targets = torch.tensor(olivetti_faces.target)

X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader([(img,lbl) for img,lbl in zip(X_train, y_train)], batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader([(img,lbl) for img,lbl in zip(X_test, y_test)], batch_size=64, shuffle=False)

num_epochs = 150


def model_train_(model_name='RNN'):
    writer = SummaryWriter()
    # 实例化模型
    model = RNN_Classifier(model_name)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
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

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar(f'training loss', loss.item(), epoch * len(train_loader) + i)
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
            writer.add_scalar(f'test accuracy', accuracy, epoch)
    writer.close()
    
model_train_('RNN')
model_train_('LSTM')
model_train_('GRU')