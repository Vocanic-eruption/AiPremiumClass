import numpy as np
#RNN 工作流程

class RNN:
    def step(self,x,hidden):
        #更新模型中隐藏层状态
        hidden=np.tanh(np.dot(self.W_hh,hidden)+np.dot(self.W_xh,x)+self.b_h)
        return hidden

rnn=RNN()
#输入序列x:[batch_size,seq_len,feature_size]或[seq_len,feature_size]
x = get_data()
seq_len=x.shape[1]
#初始化hidden_state
hidden_state=np.zeros(x.shape[2])
#循环过程中每次输入的都是一个步长的数据
#hidden_state在更新后，会在循环中和x的第i个元素一起输入到网络中
for i in range(seq_len):
    hidden_state=rnn.step(x[:,i,:],hidden_state)

#基于RNN的图像分类预测

#1.数据准备
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

ds_train=MNIST(root='data',download=True,train=True,transform=ToTensor())
dl_train=DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)

#2.模型构建
import torch
import torch.nn as nn

class MnistClassifier(nn.Module):

    def __init__(self,input_size,hidden_size,num_labels):
        super().__init__()
        #rnn layer
        self.rnn=nn.RNN(input_size=input_size,
                        hidden_size=hidden_size,
                        batch_first=True)
        #nn layer
        self.classfier=nn.Linear(in_feature=hidden_size
                                 out_size=hidden_size,
                                 batch_first=True)
        
    def forward(self,input_data):
        output,h_n=self.rnn(input_data)
        return self.classfier(h_n[0])

#3.模型训练
import torch.optim as optim
import torch.nn as nn
from tqdm as tqdm

BATCH_SIZE=16
EPOCHS=5
#optimizer,loss function
optimizer=optim.Adam(model.parameters(),le=1e-3)
criterion=nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    tpbar=tqdm(dl_train)
    for img,lbl in tpbar:
    img=img.squeeze()
    logits=model(img)
    loss=criterion(logits,lbl)
    loss.backward()
    optimizer.step()
    model.zero_grad()

    tpbar.set_description(f'epoch:{epoch+1}train_loss:{loss,item():.4f}')

#4.模型保存

#整体保存
torch.save(model.'mnist_cls.pth')

#模型参数
torch.save(model.state_dict(),'mnist_cls_state.pth')

# RNN 预测时间序列
#1.预测最后一个值
import numpy as np
def generate_time_series(batch_size,n_steps):
    freq1,freq2,offsets1,offsets2=np.random.rand(4,batch_size,1)
    time =np.linspace(0,1,nsteps)
    series =0.5*np.sin((time-offsets1)*(freg1*10+10))
    series += 0.2 *np.sin((time -offsets2)*(freq2 * 20 + 20))
    series += 0.1*(np.random.rand(batch size,n steps)-0.5)
    return series[...,np.newaxis].astype(np.float32)

n_steps=50
series =generate_time_series(10000,n_steps+1)
X_train,y_train=series[:7000,:n_steps],series[:7000，-1]
X_valid,y_valid=series[7000:9000,n_steps],series[7000:9000,-1]
X_test,y_test=series[9000:,n_steps],series[9000:,-1]

import torch
from torch.utils.data import Dataset,DataLoader

class TimeseriesDataset(Dataset):
    def __init__(self,X,y=None,train=True):
        self.X=x
        self.y=y
        self.train=train
    def __len__(self):
        return len(self.x)
    def __getitem__(self,ix):
        if self.train:
            return torch.from_numpy(self,X[ix]),torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])


dataset={
    'train':TimeSeriesDataset(X_train,y_train),
    'eval':TimeSeriesDataset(X_valid,y_valid),
    'test':TimeSeriesDataset(X_test,y_test,train=False)
}
dataloader={
    'train':DataLoader(dataset['train'],shuffle=True,batch_size=64),
    'eval':DataLoader(dataset['eval'],shuffle=False,batch_size=64),
    'test':DataLoader(dataset['test'],shuffle=False,batch_size=64)
}
        
plot_series(X_test,Y_test)


class RNN(torch.nn.Module):
    def  __init__(self):
        super().__init__()
        self.rnn=torch.nn.RNN(input_size,hidden_size=20,num_layers=1,batch_first=True)
        self.fc=torch.nn.Linear(20,1)

    def forward(self,x):
        x,h=self.rnn(x)
        y=self.fc(x[:,-1])
        return y

rnn=RNN()

from tqdm import tqdm
device="cuda"if torch.cuda.is_available() else "cpu"

def fit(model,dataloader,epochs=10):
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=torch.nn.MSELoss()
    bar=tqdm(range(1,epochs+1))
    for epoch in bar:
        model.train()
        train_loss=[]
        for batch in dataloader['train']:
            X,y=batch
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat=model(x)
            loss=criterion(y_hat,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        eval_loss=[]
        with torch.no_grad():
            for batch in dataloader['eval']:
                X,y=batch
                X,y=X.to(device),y.to(device)
                y_hat=model(X)
                loss=criterion(y_hat,y)
                eval_loss.append(loss.item())
        bar.set_description(f'loss{np.mean(train_loss):.5f}val_loss {np.mean(eval_loss):.5f}')

    def predict(model,dataloader):
        model.eval()
        with torch.no_grad():
            preds=torch.tensor([]).to(device)
            for batch in dataloader:
                X=batch
                X=X.to(device)
                pred=model(X)
                preds=torch.cat([preds,pred])
            return preds

fit(rnn,dataloader)

from sklearn.metrics import mean_squared_error

y_pred=predict(rnn,dataloader['test'])
plot_series(X_test,y_test,y_pred.cpu().numpy())
mean_squared_error(y_test,y_pred.cpu())

#使用 num_layers参数来控制循环神经网络(RNN)中的隐藏层数量
class DeepRNN(torch.nn,Module):
    def __init__(self,n_in=50,n_out=1):
        super().__init()
        self.rnn=torch.nn.RNN(input_size=1,hidden_size=20,num_layers=2,batch_first=True)
        self.fc=torch.nn.Linear(20,1)
    def forward(self,x):
        x,h=self.rnn(x)
        x=self.fc(x[:,-1])
        return x

rnn=DeepRNN()
fit(rnn,dataloader)

y_pred=predict(rnn,dataloader['test'])
plot_series(X_test,y_test,y_pred.cpu().numpy())
mean_squared_error(y_test,y_pred.cpu())

#2.预测多个值

n_steps=50
series =generate_time_series(10000,n_steps+10)
X_train,y_train=series[:7000,:n_steps],series[:7000，-10,0]
X_valid,y_valid=series[7000:9000,n_steps],series[7000:9000,-10,0]
X_test,y_test=series[9000:,:n_steps],series[9000:,-10,0]

dataset={
    'train':TimeSeriesDataset(X_train,Y_train),
    'eval':TimeSeriesDataset(X_valid,Y_valid),
    'test':TimeSeriesDataset(X_test,Y_test,train=False)
}
dataloader={
    'train':DataLoader(dataset['train'],shuffle=True,batch_size=64),
    'eval':DataLoader(dataset['eval'],shuffle=False,batch_size=64),
    'test':DataLoader(dataset['test'],shuffle=False,batch_size=64)
}
        
plot_series(X_test,Y_test)


X=X_test
for step_ahead in range(10):
    inputs=torch.from_numpy(x[:,step_ahead:]).unsqueeze(0)
    y_pred_one=predict(rnn,inputs).cpu().numpy()
    x=np.concatenate([x,y_pred_one[:,np.newaxis,:]],axis)

y_pred=X[:,n_steps:,-1]
plot_series(X_test,Y_test,y_pred)
mean_squared_error(Y_test,y_pred)

n_steps=50
X_train=series[:7000, :n_steps]
X_valid=series[7000:9000, :n_steps]
X_test=series[9000: , :n_steps]
Y=np.empty((10000,n_steps,10),dtype=np.float32)
for step_ahead in range(1,10+1):
    Y[...,step_ahead-1]=series[...,step_ahead:step_ahead+n_steps,0]
Y_train=Y[:7000]
Y_valid=Y[7000:9000]
Y_test=Y[9000:]

dataset={
    'train':TimeSeriesDataset(X_train,Y_train),
    'eval':TimeSeriesDataset(X_valid,Y_valid),
    'test':TimeSeriesDataset(X_test,Y_test,train=False)
}
dataloader={
    'train':DataLoader(dataset['train'],shuffle=True,batch_size=64),
    'eval':DataLoader(dataset['eval'],shuffle=False,batch_size=64),
    'test':DataLoader(dataset['test'],shuffle=False,batch_size=64)
}
        
class DeepRNN(torch.nn,Module):
    def __init__(self,n_out=10):
        super().__init()
        self.rnn=torch.nn.RNN(input_size=1,hidden_size=20,num_layers=2,batch_first=True)
        self.fc=torch.nn.Linear(20,n_out)

    def forward(self,x):
        x,h=self.rnn(x)
        x=self.fc(x[:,-1])
        y=self.fc(x_reshaped)
        y=y.contiguous().view(X.size(0),-1,y.size(-1))
        return y

def fit(model,dataloader,epochs=10):
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=torch.nn.MSELoss()
    bar=tqdm(range(1,epochs+1))
    for epoch in bar:
        model.train()
        train_loss=[]
        train_loss2=[]
        for batch in dataloader['train']:
            X,y=batch
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat=model(x)
            loss=criterion(y_hat,y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_loss2.append((y[:,2]-y_hat[:,-1]).pow(2).mean().item())
        model.eval()
        eval_loss=[]
        eval_loss2=[]
        with torch.no_grad():
            for batch in dataloader['eval']:
                X,y=batch
                X,y=X.to(device),y.to(device)
                y_hat=model(X)
                loss=criterion(y_hat,y)
                eval_loss.append(loss.item())
                eval_loss2.append((y[:,-1]-y_hat[:,-1]).pow(2).mean().item())
        bar.set_description(f'loss{np.mean(train_loss):.5f}',
                            f'loss_last_step{np.mean(train_loss):.5f}',
                            f'val_loss{np.mean(eval_loss):.5f}',
                            f'val_loss_last_stop{np.mean(eval_loss2):.5f}')

rnn=RNN()
fit(rnn,dataloader)

y_pred=predict(rnn,dataloader['test'])
plot_series(X_test,Y_test,y_pred[:,-1].cpu().numpy())
mean_squared_error(y_test[:,-1],y_pred[:,-1].cpu())



#1.实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。
# 要求使用tensorboard，提交代码及run目录和可视化截图。
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from sklearn.datasets import fetch_olivetti_faces
from tensorboardX import SummaryWriter
import numpy as np

# 1. 数据准备
# 使用Olivetti人脸数据集
faces = fetch_olivetti_faces()
X = faces.images
y = faces.target

# 将数据转换为PyTorch张量
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# 划分训练集和测试集
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
X_train, X_test = torch.split(X, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])

# 定义数据加载器
BATCH_SIZE = 16
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 模型构建
class FaceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, rnn_type='RNN'):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == 'BiRNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            hidden_size *= 2

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_data):
        input_data = input_data.view(input_data.size(0), -1, input_data.size(-1))
        if self.rnn_type in ['LSTM']:
            output, (h_n, _) = self.rnn(input_data)
        else:
            output, h_n = self.rnn(input_data)

        if self.rnn_type == 'BiRNN':
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            h_n = h_n.squeeze(0)

        return self.classifier(h_n)

# 3. 模型训练和评估
EPOCHS = 5
writer = SummaryWriter('runs')

rnn_types = ['RNN', 'LSTM', 'GRU', 'BiRNN']
for rnn_type in rnn_types:
    model = FaceClassifier(input_size=X_train.size(-1), hidden_size=128, num_labels=len(np.unique(y)), rnn_type=rnn_type)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        tpbar = tqdm(train_loader)
        for img, lbl in tpbar:
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += lbl.size(0)
            train_correct += predicted.eq(lbl).sum().item()

            tpbar.set_description(f'{rnn_type} epoch:{epoch+1} train_loss:{loss.item():.4f}')

        train_accuracy = 100. * train_correct / total
        writer.add_scalar(f'{rnn_type}/Train Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar(f'{rnn_type}/Train Accuracy', train_accuracy, epoch)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for img, lbl in test_loader:
                logits = model(img)
                loss = criterion(logits, lbl)
                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += lbl.size(0)
                test_correct += predicted.eq(lbl).sum().item()

        test_accuracy = 100. * test_correct / total
        writer.add_scalar(f'{rnn_type}/Test Loss', test_loss / len(test_loader), epoch)
        writer.add_scalar(f'{rnn_type}/Test Accuracy', test_accuracy, epoch)

writer.close()

# 4. 模型保存
# 整体保存
torch.save(model, 'face_cls.pth')
# 模型参数
torch.save(model.state_dict(), 'face_cls_state.pth')


#2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#   数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 加载数据集
url = 'https://www.kaggle.com/datasets/smid80/weatherww2'
# 这里假设数据集已经下载到本地并命名为weather.csv
data = pd.read_csv('weather.csv')
# 提取最高气温列
max_temperature = data['Max TemperatureC'].values

# 数据预处理
def generate_time_series(data, batch_size, n_steps):
    series = []
    for _ in range(batch_size):
        start_idx = np.random.randint(len(data) - n_steps - 1)
        series.append(data[start_idx:start_idx + n_steps + 1])
    series = np.array(series)[:, :, np.newaxis].astype(np.float32)
    return series

n_steps = 50
# 生成训练数据
series = generate_time_series(max_temperature, 10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# 定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        if self.train:
            return torch.from_numpy(self.X[ix]), torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])

# 定义数据加载器
dataset = {
    'train': TimeSeriesDataset(X_train, y_train),
    'eval': TimeSeriesDataset(X_valid, y_valid),
    'test': TimeSeriesDataset(X_test, y_test, train=False)
}
dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.fc(x[:, -1])
        return y

# 训练模型
def fit(model, dataloader, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    writer = SummaryWriter('runs')

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for X, y in dataloader['train']:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader['train'])
        writer.add_scalar('Train Loss', train_loss, epoch)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X, y in dataloader['eval']:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y.unsqueeze(-1))
                valid_loss += loss.item()

        valid_loss /= len(dataloader['eval'])
        writer.add_scalar('Valid Loss', valid_loss, epoch)

    writer.close()
    return model

# 训练预测1天最高气温的模型
rnn = RNN()
rnn = fit(rnn, dataloader, epochs=10)

# 预测1天最高气温
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
y_pred_1day = []
with torch.no_grad():
    for X in dataloader['test']:
        X = X.to(device)
        y_pred = model(X)
        y_pred_1day.extend(y_pred.cpu().numpy())

y_pred_1day = np.array(y_pred_1day).flatten()
mse_1day = mean_squared_error(y_test, y_pred_1day)
print(f"1天最高气温预测的均方误差: {mse_1day}")

# 预测连续5天最高气温
n_steps = 50
# 生成训练数据
series = generate_time_series(max_temperature, 10000, n_steps + 5)
X_train, y_train = series[:7000, :n_steps], series[:7000, -5:]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -5:]
X_test, y_test = series[9000:, :n_steps], series[9000:, -5:]

# 定义数据集类
dataset = {
    'train': TimeSeriesDataset(X_train, y_train),
    'eval': TimeSeriesDataset(X_valid, y_valid),
    'test': TimeSeriesDataset(X_test, y_test, train=False)
}
dataloader = {
    'train': DataLoader(dataset['train'], shuffle=True, batch_size=64),
    'eval': DataLoader(dataset['eval'], shuffle=False, batch_size=64),
    'test': DataLoader(dataset['test'], shuffle=False, batch_size=64)
}

# 定义RNN模型
class RNN_5days(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = nn.Linear(20, 5)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.fc(x[:, -1])
        return y

# 训练模型
rnn_5days = RNN_5days()
rnn_5days = fit(rnn_5days, dataloader, epochs=10)

# 预测连续5天最高气温
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
y_pred_5days = []
with torch.no_grad():
    for X in dataloader['test']:
        X = X.to(device)
        y_pred = model(X)
        y_pred_5days.extend(y_pred.cpu().numpy())

y_pred_5days = np.array(y_pred_5days)
mse_5days = mean_squared_error(y_test, y_pred_5days)
print(f"连续5天最高气温预测的均方误差: {mse_5days}")

# 可视化
plt.plot(y_test[:, 0], label='True 1-day Max Temperature')
plt.plot(y_pred_1day, label='Predicted 1-day Max Temperature')
plt.xlabel('Time')
plt.ylabel('Max Temperature (Celsius)')
plt.title('1-day Max Temperature Prediction')
plt.legend()
plt.show()

for i in range(5):
    plt.plot(y_test[:, i], label=f'True {i+1}-th day Max Temperature')
    plt.plot(y_pred_5days[:, i], label=f'Predicted {i+1}-th day Max Temperature')
    plt.xlabel('Time')
    plt.ylabel('Max Temperature (Celsius)')
    plt.title(f'{i+1}-th day Max Temperature Prediction in 5-day Forecast')
    plt.legend()
    plt.show()
