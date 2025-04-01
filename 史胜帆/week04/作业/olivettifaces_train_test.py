#fetch_olivetti_face （400，4090） 40类
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader,TensorDataset #数据加载器batch_size
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #此次数据集需要做切分

olivetti_faces = fetch_olivetti_faces(data_home='./olivetti_faces_data',shuffle = True) # 下载后是.pkz文件 用.data .target即可查看

#看看形状和样子
print(olivetti_faces) 
print(olivetti_faces.data)
print(olivetti_faces.data.shape)
print(olivetti_faces.target)
print(type(olivetti_faces.target)) # list没有.shape numpy 和torch pandas有
print(olivetti_faces.target.shape)

labels = [set(clazz for clazz in olivetti_faces.target)]
print(labels)

# face = olivetti_faces.images[0]
# clazz = olivetti_faces.target[0]
# plt.imshow(face,cmap = 'gray')
# plt.title(clazz)
# plt.show()

#开始训练

# 1 参数设置
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 100
BATCH_SIZE2 = 10

# 2 数据预处理 包括上面和现在的batch_size
X_train,x_test,y_train,y_test = train_test_split(olivetti_faces.data,olivetti_faces.target,test_size = 0.2,stratify = olivetti_faces.target,random_state = 35)
X_train = torch.FloatTensor(X_train)  #DataLoder自带tensor转换
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)
train_datasets = TensorDataset(X_train,y_train) #用于data和target分开的数据集 二者合并&转为tensor
test_datasets = TensorDataset(x_test,y_test)
train_dl = DataLoader(train_datasets,batch_size = BATCH_SIZE,shuffle = True)





# 3 定义网络层、网络结构 和 前向运算 以类的方式定义 使网络层更加灵活
class Model(nn.Module):  # 初始 shape(400,4090)
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features = 4096,out_features = 2048)
        self.norm = nn.BatchNorm1d(2048)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.1)
        self.linear2 = nn.Linear(in_features = 2048,out_features = 1024)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p = 0.2)
        self.linear3 = nn.Linear(in_features = 1024,out_features = 512)
        self.norm1 = nn.BatchNorm1d(512)
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p = 0.35)
        self.linear4 = nn.Linear(in_features = 512,out_features = 40)
        
        self.activation4 = nn.ReLU() # 为什么用softmax反而训练效果不如以前 特征消失？
    
    def forward(self,X):
        
        out = self.linear1(X)
        out = self.norm(out)
        out = self.activation1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.activation2(out)
        out = self.dropout2(out)
        out = self.linear3(out)
        out = self.norm1(out)
        out = self.activation3(out)
        out = self.dropout3(out)
        out = self.linear4(out)
        
        out = self.activation4(out)

        return out

model = Model()

# 4 选择损失函数

loss_fn = nn.CrossEntropyLoss()

# 5 选择优化器 参数更新
#print(model)
optimizer = torch.optim.Adam(model.parameters(),lr = LR)
#print(par  for par in model.parameters())

# 6 开始训练

for epoch in range(EPOCHS):
    model.train()
    for data,target in train_dl: # 用zip(X_train,y_train)需要注意转为张量 注意数据类型和形状
        #target = target.long()
        #前向运算
        y_hat = model(data)
        #计算损失
        loss_val = loss_fn(y_hat,target)
        #计算梯度
        optimizer.zero_grad()
        loss_val.backward()
        #梯度下降 参数更新
        optimizer.step()
    
    print(f'epoch:{epoch},loss:{loss_val.item()}')


#模型推理 预测
test_dl = DataLoader(test_datasets,batch_size = BATCH_SIZE2)

with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for data,target in test_dl:
        predict = model(data)
        _,predicted = torch.max(predict,dim = 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(predicted)
    print(target)
    print(f'acc:{(correct / total) * 100}%')

        




