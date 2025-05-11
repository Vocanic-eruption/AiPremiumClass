# 1、导入包
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2、数据准备
data_path = os.path.join(os.path.dirname(__file__), "./face_data")
olivetti_faces = fetch_olivetti_faces(data_home=data_path, shuffle=True)
print(olivetti_faces.data.shape)
print(olivetti_faces.target.shape)
print(olivetti_faces.images.shape)
batch_size = 50


# 转换为 PyTorch 张量
# olivetti_faces.images = torch.tensor(olivetti_faces.images, dtype=torch.float)
# olivetti_faces.target = torch.tensor(olivetti_faces.target, dtype=torch.long)
olivetti_faces.images = torch.FloatTensor(olivetti_faces.images)
olivetti_faces.target = torch.LongTensor(olivetti_faces.target)

# 数据集划分
# x_train, x_test, y_train, y_test = train_test_split(olivetti_faces.images, olivetti_faces.target, test_size=0.25, random_state=42)
x_train = olivetti_faces.images[:300]
y_train = olivetti_faces.target[:300]
x_test = olivetti_faces.images[300:]
y_test = olivetti_faces.target[300:] 

train_dataset = list(zip(x_train, y_train))
test_dataset = list(zip(x_test, y_test))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 3、模型构建
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type='lstm'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first=True表示输入和输出数据的形状为(batch_size, seq_len, input_size)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type:{rnn_type}")
        
        self.fc = nn.Linear(hidden_size, num_classes) 
        
    def forward(self, x):
        if self.rnn_type == 'lstm':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, h0)

        return self.fc(out[:, -1, :])
    
# 4、训练模型函数
def train_model(train_loader, test_loader, model, cross_entropy, optimizer, num_epochs, writer, model_name):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i,(inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 50 == 0:
                print('Epoch %d, Step %d, Loss: %.4f, Accuracy: %.2f%%' % (epoch+1, i+1, running_loss/50, 100*correct/total))
                writer.add_scalar('Loss/train_%s' % model_name, running_loss/50, epoch*num_epochs + i)
                writer.add_scalar('Accuracy/train_%s' % model_name, 100*correct/total, epoch*num_epochs + i)
        model.eval()
        for i,(inputs, labels) in enumerate(test_loader):
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            if i % 50 == 0:
                print('Epoch %d, Step %d, Test Accuracy: %.2f%%' % (epoch+1, i+1, 100*test_correct/test_total))
                writer.add_scalar('Accuracy/test_%s' % model_name, 100*test_correct/test_total, epoch*num_epochs + i)

# 5、初始化超参数
lr = 0.001
num_epochs = 10
input_size = 64
output_size = 40
hidden_size = 256
num_layers = 3
model_list = ['rnn', 'lstm', 'gru']
writer = SummaryWriter()

#6、训练
for rnn_type in model_list:
    print("--------------------------------------------------------------------")
    model = RNNModel(input_size, hidden_size,  num_layers, output_size, rnn_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    cross_entropy = nn.CrossEntropyLoss()
    train_model(train_dataloader, test_dataloader, model, cross_entropy, optimizer, num_epochs, writer, rnn_type)
    writer.close()
