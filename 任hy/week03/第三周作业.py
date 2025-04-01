import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import KMNIST
from torchvision.transforms.v2 import ToTensor
from torch.utils.data import DataLoader

# 超参数
LR = [0.01,0.05,0.1]
epochs = [20,30,40]
BATCH_SIZE = 64

train_data = KMNIST(root='./KMINST_data.',train=True,download=True,transform=ToTensor())
test_data = KMNIST(root='./KMINST_data',train=False,download=True,transform=ToTensor())

train_dl = DataLoader(train_data,batch_size = BATCH_SIZE,shuffle=True)

# 拿到种类
# labels = set(clz for name,clz in train_data)
# print(labels) 

model1 = nn.Sequential(
    nn.Linear(784,64),
    nn.Sigmoid(),
    nn.Linear(64,10)
)

model2 = nn.Sequential( # 更改神经元数量
    nn.Linear(784,80),
    nn.Sigmoid(),
    nn.Linear(80,10)
)

model3 = nn.Sequential( # 更改隐藏层数量
    nn.Linear(784,128),
    nn.Sigmoid(),
    nn.Linear(128,32),
    nn.Sigmoid(),
    nn.Linear(32,10)
)

def get_model(model_id):
    if model_id == 0:
        return model1
    elif model_id == 1:
        return model2
    elif model_id == 2:
        return model3
    else:
        raise ValueError("数字有误!")

loss_fn = nn.CrossEntropyLoss()
print("输入三个取值范围是1-3的数字,分别代表LR,epochs和model")

while True:
    s = input("输入三个数，空格隔开")
    a, b, c = map(int, s.split())
    if any(value < 1 or value > 3 for value in [a, b, c]):
        raise ValueError("输入错误：至少有一个值不在 1~3 范围内！")
    LR_test = LR[a-1]
    epochs_test = epochs[b-1]
    model = get_model(c-1)
    
    optimizer = torch.optim.SGD(model.parameters(),lr=LR_test)


    for epoch in range(epochs_test):
        for data,target in train_dl:
            output = model(data.reshape(-1,784))
            loss = loss_fn(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Epoch:{epoch},Loss:{loss.item()}')


    # 测试
    test_dl = DataLoader(test_data,batch_size=BATCH_SIZE)
    correct = 0
    total = 0
    with torch.no_grad():
        for data,target in test_dl:
            output = model(data.reshape(-1,784))
            items,predicts = torch.max(output,1) # 沿着列查找最大值
            total+=target.shape[0]
            correct+=(predicts == target).sum().item()
    print(f'在学习率:{LR_test}下选择模型{c-1},训练:{epochs_test}次,准确率:{correct/total*100}%')

