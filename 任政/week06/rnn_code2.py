import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):

    def __init__(self,):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=28,   # x的特征维度
            hidden_size=50,  # 隐藏层神经元数量 w_ht[50,28], w_hh[50,50]
            bias=True,        # 偏置[50]
            num_layers=5,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.fc = nn.Linear(50, 10)  # 输出层 

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out


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
    model = RNN_Classifier()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
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


    # 保存全部
    torch.save(model, 'rnn_model.pth')
    # 保存模型参数
    torch.save(model.state_dict(), 'rnn_model_params.pth')

    writer.close()



    # 加载模型
    model = torch.load('rnn_model.pth')

    # 加载模型参数
    model = RNN_Classifier()
    model.load_state_dict(torch.load('rnn_model_params.pth'))
