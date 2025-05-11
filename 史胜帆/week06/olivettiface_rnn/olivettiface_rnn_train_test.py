# 使用fetch_olivetti_faces数据集 观察RNN、BiRNN、GRU、LSTM训练的效果差异 并用tensorboard展示出来
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

datasets = fetch_olivetti_faces(data_home = './olivetti_faces_data')
print(datasets.data.shape)
print(datasets.target.shape)

# 模型参数
EPOCHES = 50
LR = 0.001
BATCH_SIZE_TRAIN = 40  # 分别尝试了 10 40 80 100 发现batch_size的大小不是越大越好 也不是越小越好 模型是对一个batch求梯度的 太大不易拟合 太小不稳定&速度慢
BATCH_SIZE_TEST = 10

# 数据预处理
data = torch.FloatTensor(datasets.data)
target = torch.LongTensor(datasets.target)
X_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.2,stratify = target,random_state = 35)
train_datasets = TensorDataset(X_train,y_train) #对tensor作用
test_datasets = TensorDataset(x_test,y_test)
train_dl = DataLoader(train_datasets,batch_size = BATCH_SIZE_TRAIN,shuffle = True)
test_dl = DataLoader(test_datasets,batch_size = BATCH_SIZE_TEST,shuffle = False)

# 模型定义
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size = 64,hidden_size = 128,bias = True,batch_first = True,num_layers = 1,dropout = 0.2)
        #比对 RNN GRU LSTM BiRNN(怎么用？) 发现在相同条件下 效果 RNN<GRU<LSTM
        self.norm = nn.BatchNorm1d(128)
        self.fc = nn.Linear(in_features = 128,out_features = 40,bias = True)
        self.act = nn.ReLU()
    
    def forward(self,X):
        out,_ = self.rnn(X)
        out = self.norm(out[:,-1,:])
        out = self.fc(out)
        out = self.act(out)
        
        return out
    
#创建模型
model = Model()
# 损失函数
cerition = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr = LR)

# 模型训练
writer = SummaryWriter()

for epoch in range(EPOCHES):
    model.train()
    for i,(data,target) in enumerate(train_dl):
        # 每次计算前清零梯度 防止梯度累加
        optimizer.zero_grad()
        output = model(data.reshape(-1,64,64))
        # 损失计算
        loss = cerition(output,target)
        # 反向传播 计算梯度
        loss.backward()
        # 梯度下降 更新参数
        optimizer.step()
        if i % 100 == 0:
            print(f'epoch:{epoch + 1}/{EPOCHES},loss:{loss.item()}')
            writer.add_scalar('train_loss',loss.item(),epoch*(len(train_dl)) + i)
# 模型推导
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for data,target in test_dl:
            output = model(data.reshape(-1,64,64))
            _,predicted = torch.max(output,dim = 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
        acc = correct / total * 100
        print(f'acc:{acc:4f}%')
        writer.add_scalar('accuarcy',acc,epoch)
        print(predicted)
        print(target)

writer.close()

