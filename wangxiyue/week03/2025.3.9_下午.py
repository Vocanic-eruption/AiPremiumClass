import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



##################预习内容##########################
train_data = datasets.FashionMNIST(
    root='../data',
    train=True, # 训练集
    download=True,
    transform=transforms.ToTensor()  # 原始数据转换成 张量
)

test_data = datasets.FashionMNIST(
    root='../data',
    train=False, # 测试集
    download=True,
    transform=transforms.ToTensor()
)



#每个批次大小
batch_size = 64

#数据加载器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# img_data = datasets.FashionMNIST(
#     root='../data',
#     train=True, # 训练集
#     download=True,
# )
# img,clzz = img_data[12000]
# print(img)
# plt.imshow(img,cmap='gray') # img是一个PIL.Image对象(Python原始数据) ,  !!! transform=transforms.ToTensor()
# plt.show()

labels =  set([clz for  img, clz in train_data])
print(labels)



# 获取可用设备 cpu 或gpu , m1
if(torch.backends.mps.is_available()):
    device = torch.device('mps')
elif(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'use {device} ')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 展平 张量
        # Sequential 顺序模型
        self.linear_relu_stack = nn.Sequential(
            # wx + b = [61,1,784] * [784 , 512] = 64,1,512
            # 线性层， nn.Linear(输入特征数量,输出特征数量，是否设置bias偏置：偏置数量与输出特征数量相等 )
            nn.Linear(28 * 28, 512),
            #（0，X），ReLU有效缓解梯度消失、 加速训练
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

input_image = torch.randn(3,28,28)
print(input_image.size())

flatten = nn.Flatten();
flat_image = flatten(input_image)
print('flat_image',flat_image.size())

#定义线性层
layer1 = nn.Linear(in_features=28*28,out_features = 20)
hidden1 = layer1(flat_image)

print(f'before ReLU : {hidden1} \n\n') # 包括 正负
hidden1 = nn.ReLU()(hidden1)
print(f'after ReLU : {hidden1} \n\n') # 只有 0 和 正

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,20)
)
input_image = torch.randn(3,28,28)
logits = seq_modules(input_image)
print(logits)


