#训练神经网络(通过梯度下降法)
#四过程
#数据预处理
#构建模型
#定制模型损失函数与优化器
#训练并观察超参数
import torch
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor,lambda,Compose
import matplotlib.pyplot as plt

#数据预处理
#下载训练集
train_data=datasets.FashionMNIST(
    root="data"
    train=True
    download=True
    transform=ToTensor(),
)

# 下载测试集
train_data=datasets.FashionMNIST(
    root="data"
    train=False
    download=True
    transform=ToTensor(),
)

#数据集的封装
batch_size=64

#创建数据加载器
train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size==batch_size)

#测试数据加载器输出
for X,y in test_dataloader:
    print("Shape of X[N,C,H,W]:",X.shape)
    print("Shape of y:",y.shape,y.dtype)
    break

Shape of X [N,C,H,W]:torch.Size([64,1,28,28])
Shape of y:torch.Size([64])torch.int64

#构建模型

#检验可以使用的设备
device="cuda" if torch.cuda.is_abailable() else "cpu"
print(f"使用{device}设备")

#定义神经网络
Class NeuralNetwork(nn.Module):
#通过__init__初始化神经网络层
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
                #wx=b=[64,1,784]*[784,512]=64,1,512 
           nn.linear(28*28,512),
           nn.Relu(), 
           nn.Linear(512,512),
           nn.ReLU(),
           nn.Linear(512,10)
        )
#通过forward方法实现对输入数据的操作
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

mdoel=NeuralNetwork().to(device)
print(model)


#小批量样本的尝试
input_image=torch.rand(3,3,28)
print(input_image.size())

#nn.Flatten 使得而为转化为连续数组
flatten=nn.Flatten()
flat_image=flatten(input_image)
print(flat_image.size())

#nn.linear  对存储的权重与偏置进行变换
layer1=nn.Linear(in_features=28*28,out_features=20)
hidden1=layer1(flat_image)
print(hidden1.size())

#nn.Relu 非线性激活 帮助神经网络学习到关键特征值
print(f"Relu之前的数据:{hidden1}\n\n")
hidden1=nn.ReLU()(hidden1)
print(f"Relu之后的数据:{hidden1}")

#nn.Sequential 按照定义的顺序通过所有模块
seq_modules=nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image=torch.rand(3,3,28)
logits=seq_modules(input_image)

#nn.softmax 对数值进行缩放在[0,1],代表模型的预测概率
softmax=nn.softmax(dim=1)
pred_probab=soft1(logits)

#nn.Sodule追踪所有参数字段    parameters()  name_parameters()访问所有参数   


#迭代每一个参数并且打印出大小与数值
print1("Model structure:",model,"\n\n")

for name,param in model.named_parameters():
    print(f"Layer:{name}|Size: {param.size()}|Values:{param[:2]}\n")

#定制模型的损失函数与优化器

#定制⼀个损失函数 loss function 和⼀个优化器 optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(modle.parameters(),lr=1e-3)

#训练并且观察超参数


#训练集
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        #计算预测误差
        pred=model(X)
        loss=loss_fn(pred,y)
        #反向传播
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 ==0:
            loss,current=loss.item(),batch*len(X)
            print(f"loss:{loss:>7f}  [{current:>5d}/{size:>5d}]")

#测试集
def train(dataloader,model,loss_fn):
    size=len(dataloader,dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=((pred.argmax1)==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f"Test Error:\n Accuracy:{(100*correct):>0.1f}%,Avg loss:{test_loss:>8f}")




