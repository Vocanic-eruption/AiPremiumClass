                                                      # 搭建的神经网络，使用olivettiface数据集进行训练。
from sklearn.datasets import fetch_olivetti_faces
olivetti_faces=fetch_olivetti_faces(data_home='./face_data',shuffle=True)
print(olivetti_faces.data.shape)
print(olivetti_faces.target.shape)
print(olivetti_faces.images.shape)
import matplotlib.pyplot as plt

face=olivetti_faces.images[1]
plt.imshow(face,cmap='gray')
plt.show()
olivetti_faces.data[1]
olivetti_faces.target

import torch
import torch.nn as nn
images=torch.tensor(olivetti_faces.data)
targets=torch.tensor(olivetti_faces.target)
images.shape
targets.shape
dataset=[(img,lbl)for img,lbl  in zip(images,targets)]
dataset[0]
dataloader=torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=True)
#device=torch.device('mps'if torch.backends.mps.is_available()else'cpu')
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
#device='cpu' 

model=nn.Sequential(
    nn.Linear(4096,8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(),
    nn.Linear(8192,16384),
    nn.BatchNorm1d(16384),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(16384,1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(1024,40),
).to(device)
print(model)
                                       # 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
###########################归一化#################################
import torch
import torch.nn as nn
# 定义⼀个简单的全连接⽹络，包含 BatchNorm 层
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# 创建模型实例
model = SimpleNet()
###############################正则化############################
import torch
import torch.nn as nn
class DropoutNet(nn.Module):
    def __init__(self):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(p=0.5) # 随机失活50%的神经元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = DropoutNet()
                                      #  尝试不同optimizer对模型进行训练，观察对比loss结果
#############################1.数据预处理########################################
#下载训练集
training_data=datasets.KMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试集
test_data=datasets.KMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#数据集的封装
batch_size=64

#创建数据加载器
train_dataloader=DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

#测试数据加载器输出
for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]:", X.shape)
    print("Shape of y:", y.shape, y.dtype)
    break
###################################2.构建模型#####################################

#检验可以使用的设备
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"使用{device}设备")

#定义神经网络
class NeuralNetwork(nn.Module):
#通过__init__初始化神经网络层
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
                #wx=b=[64,1,784]*[784,512]=64,1,512 
            nn.Linear(28*28,512),
            nn.ReLU(),  
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
#通过forward方法实现对输入数据的操作
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NeuralNetwork().to(device)
print(model)


#########################3.定制模型的损失函数与优化器##############################

#定制⼀个损失函数 loss function 和⼀个优化器 optimizer
loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)



#############################4.训练并且观察超参数###################################

#训练集       进行预测-反向传播-调整模型参数
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()      #进入训练模型
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        #计算预测的误差
        pred=model(X)
        loss=loss_fn(pred,y)
        
        #反向传播
        model.zero_grad()  #重置模型中的参数的梯度值为0
        loss.backward()   #计算梯度
        optimizer.step()    #更新梯度值

        if batch % 100 ==0:
            loss,current=loss.item(),batch * len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")



#测试集    检查模型性能 确保优化效果
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=((pred.argmax(dim=1))==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}")

#多轮迭代训练
epochs=40
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("训练完成！")


#############################1.数据预处理########################################
#下载训练集
training_data=datasets.KMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试集
test_data=datasets.KMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#数据集的封装
batch_size=64

#创建数据加载器
train_dataloader=DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

#测试数据加载器输出
for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]:", X.shape)
    print("Shape of y:", y.shape, y.dtype)
    break
###################################2.构建模型#####################################

#检验可以使用的设备
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"使用{device}设备")

#定义神经网络
class NeuralNetwork(nn.Module):
#通过__init__初始化神经网络层
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
                #wx=b=[64,1,784]*[784,512]=64,1,512 
            nn.Linear(28*28,512),
            nn.ReLU(),  
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
#通过forward方法实现对输入数据的操作
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NeuralNetwork().to(device)
print(model)


#########################3.定制模型的损失函数与优化器##############################

#定制⼀个损失函数 loss function 和⼀个优化器 optimizer
loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)



#############################4.训练并且观察超参数###################################

#训练集       进行预测-反向传播-调整模型参数
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()      #进入训练模型
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        #计算预测的误差
        pred=model(X)
        loss=loss_fn(pred,y)
        
        #反向传播
        model.zero_grad()  #重置模型中的参数的梯度值为0
        loss.backward()   #计算梯度
        optimizer.step()    #更新梯度值

        if batch % 100 ==0:
            loss,current=loss.item(),batch * len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")



#测试集    检查模型性能 确保优化效果
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=((pred.argmax(dim=1))==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}")

#多轮迭代训练
epochs=40
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("训练完成！")




#############################1.数据预处理########################################
#下载训练集
training_data=datasets.KMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试集
test_data=datasets.KMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#数据集的封装
batch_size=64

#创建数据加载器
train_dataloader=DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=batch_size)

#测试数据加载器输出
for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]:", X.shape)
    print("Shape of y:", y.shape, y.dtype)
    break
###################################2.构建模型#####################################

#检验可以使用的设备
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"使用{device}设备")

#定义神经网络
class NeuralNetwork(nn.Module):
#通过__init__初始化神经网络层
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
                #wx=b=[64,1,784]*[784,512]=64,1,512 
            nn.Linear(28*28,512),
            nn.ReLU(),  
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
#通过forward方法实现对输入数据的操作
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NeuralNetwork().to(device)
print(model)


#########################3.定制模型的损失函数与优化器##############################

#定制⼀个损失函数 loss function 和⼀个优化器 optimizer
loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)



#############################4.训练并且观察超参数###################################

#训练集       进行预测-反向传播-调整模型参数
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()      #进入训练模型
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        #计算预测的误差
        pred=model(X)
        loss=loss_fn(pred,y)
        
        #反向传播
        model.zero_grad()  #重置模型中的参数的梯度值为0
        loss.backward()   #计算梯度
        optimizer.step()    #更新梯度值

        if batch % 100 ==0:
            loss,current=loss.item(),batch * len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")



#测试集    检查模型性能 确保优化效果
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=((pred.argmax(dim=1))==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}")

#多轮迭代训练
epochs=40
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("训练完成！")
