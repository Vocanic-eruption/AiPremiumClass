import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
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
        # 取最后一个时间点的输出值，当且仅当num_layers=1时outputs[:,-1,:]与l_h[0]相同
        out = self.fc(outputs[:,-1,:])
        return out
if __name__=='__main__':
    
    writer=SummaryWriter()
    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    train_dataset=MNIST(root='/pytorch_class/week6/data',download=True,train=True,transform=ToTensor())
    test_dataset=MNIST(root='/pytorch_class/week6/data',download=True,train=False,transform=ToTensor())

    #创建数据加载器
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

    #创建分类模型
    model=RNN_Classifier()
    model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    #训练模型
    num_epochs=10
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i,(img,labels)in enumerate(train_loader):
            img,labels=img.to(device),labels.to(device)
            optimizer.zero_grad()
            #这里时将维度为1的通道那一维给压缩掉，使之复合RNN的输入维度 B T F
            output=model(img.squeeze())
            loss=criterion(output,labels)
            loss.backward()
            #梯度裁剪防止RNN梯度爆炸设置范数为1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            if i%100==0:
                print(f'Epoch[{epoch+1/num_epochs}],Loss:{loss.item():4f}')
                #使用tensorboard
                writer.add_scalar('training loss/RNN',loss.item(),epoch*(len(train_loader)+i))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images,labels in test_loader:
                images,labels = images.to(device),labels.to(device)
                output=model(images.squeeze())
                #只关注预测下标
                _,predicted=torch.max(output.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
            writer.add_scalar('test accuracy/RNN', accuracy, epoch)

   
