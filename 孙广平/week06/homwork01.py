
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNNModel(nn.Module):
    def __init__(self, rnn_type='RNN'):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = 28
        self.hidden_size = 128
        self.bias = True
        self.num_layers = 5
        self.num_classes = 10
        
        # RNN, LSTM, GRU
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

        
        # 全连接层
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, time_steps, input_size] => [batch_size, 28, 28]
        out, _ = self.rnn(x)  # 获取 RNN 输出
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images.squeeze())
        # 计算损失
        loss = criterion(outputs, labels)
        # 清零梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch}] Training Loss: {avg_loss:.4f}')
    writer.add_scalar('Training Loss', avg_loss, epoch)
        
        


def test(model, test_loader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.squeeze())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    writer.add_scalar('Test Accuracy', accuracy, epoch)
    


if __name__ == '__main__':

    writer = SummaryWriter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 实例化模型
    # model = RNNModel(rnn_type='RNN')  
    # model = RNNModel(rnn_type='LSTM') 
    model = RNNModel(rnn_type='GRU') 
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device,epoch)
        test(model, test_loader, device,epoch)
        

    writer.close()

    # 保存模型
    # torch.save(model.state_dict(), 'rnn_model.pth')
    # torch.save(model.state_dict(), 'LSTM_model.pth')
    torch.save(model.state_dict(), 'GRU_model.pth')
    
