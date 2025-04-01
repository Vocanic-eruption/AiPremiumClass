# 导入必要包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量
from torch.utils.data import DataLoader  # 数据加载器
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt


#定义模型
class modele(nn.Module) :
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64*64, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 40)
        #self.BatchNorm1d = nn.BatchNorm1d(2048)
        #self.Dropout = nn.Dropout()
        #
        #self.linear = nn.Linear()
        #self.BatchNorm1d = nn.BatchNorm1d()
        #self.Dropout = nn.Dropout()
        #self.relu = nn.ReLU()

    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        final = self.linear4(out)
        return final




#绘制训练损失函数
def draw_train_hist(hist):
    plt.plot(hist)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
#数据加载函数
def load_data(data_home,batch_size):

    olivetti_faces = fetch_olivetti_faces(data_home=data_home, shuffle=True)
    datas = torch.tensor(olivetti_faces.data)
    datas = datas.view(-1, 64*64)  # 展平为一维
    targets = torch.tensor(olivetti_faces.target)
    dataset = [(img,lbl) for img,lbl in zip(datas, targets)]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    return dataloader

#定义超参数
model = modele()

LR = 0.001
epochs = 50
batch_size = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义损失函数
loss_fn = nn.CrossEntropyLoss()
#定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

#加载数据
train_dl = load_data(data_home='./face_data',batch_size = batch_size)

model.to(device)  # 将模型移动到设备上
model.train()  # 正则化&归一化生效
model_hist = {'loss': []}
def train_iter(epochs, model, train_dl):
    for epoch in range(epochs):
        epoch_loss = 0.0  # 记录每个epoch的损失
        for data, target in train_dl:
            data, target = data.to(device), target.to(device)  # 将数据移动到设备上
            output = model(data)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 累加每个batch的损失
        epoch_loss /= len(train_dl)  # 计算平均损失
        model_hist['loss'].append(epoch_loss)  # 记录每个epoch的损失
        print(f'Epoch:{epoch} Loss: {epoch_loss}')
    return model_hist['loss']

#绘制训练损失函数
def draw_train_hist(hist):
    plt.plot(hist)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')  # 保存图像到文件
    print("训练损失图已保存为 training_loss.png")
#开始训练
model_hist = train_iter(epochs, model, train_dl)
draw_train_hist(model_hist)


