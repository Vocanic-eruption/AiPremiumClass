# 1. 使用pytorch搭建神经网络模型，实现对KMNIST数据集的训练。
# https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST
# 2. 尝试调整模型结构（变更神经元数量，增加隐藏层）来提升模型预测的准确率
# 3. 调试超参数，观察学习率和批次大小对训练的影响。

import torch
import torch.nn as nn
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader  # 数据加载器
from torchvision.transforms.v2 import ToTensor     # 转换图像数据为张量

device = None
if torch.cuda.is_available():  # 判断是否有可用的GPU
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is unavailable")
    device = torch.device("cpu")

# 超参数设置
learning_rate = 0.0005
epochs = 100
batch_size = 20
# 训练数据集加载
train_data = KMNIST(root='./kuzushiji_data', train=True, download=True, 
                          transform=ToTensor())
test_data = KMNIST(root='./kuzushiji_data', train=False, download=True,
                         transform=ToTensor())
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # shuffle=True表示打乱数据

# print(train_data.data.shape) #torch.Size([60000, 28, 28])

# 模型定义
myModel = nn.Sequential(
    nn.Linear(784, 784), # 输入层到隐藏层的线性变换  784个输入层，128个隐藏层
    nn.Sigmoid(), # 隐藏层的Sigmoid激活函数
    nn.Linear(784, 28) # 隐藏层到输出层的线性变换  128个隐藏层，28个输出层
).to(device)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for data, target in train_dl:
        input_data = data.reshape(-1, 784).to(device) #torch.Size([128, 784]) 转换形状，变成128个样本，每个样本是一个长度为784的向量，转换目的是为了匹配模型的输入层
        target = target.to(device) # 确保标签也在GPU上
        output = myModel(input_data)
        loss = loss_fn(output, target) # 计算损失
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        # print('epoch: ', epoch, 'loss: ', loss.item())
        
    print('epoch: ', epoch, 'loss: ', loss.item())
    
# 测试
test_dl = DataLoader(test_data, batch_size=batch_size)
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for data, target in test_dl:
        myModel = myModel.to("cpu")
        output = myModel(data.reshape(-1, 784))
        _, predicted = torch.max(output, 1)  # 返回每行最大值和索引
        total += target.size(0)  # size(0) 等效 shape[0]
        correct += (predicted == target).sum().item()

print(f'Accuracy: {correct/total*100}%')
    
# 观察  隐藏层神经元 300，输出层神经元 28
# epoch:  0 loss:  2.4472904205322266
# epoch:  1 loss:  2.3437845706939697
# epoch:  2 loss:  2.2885191440582275
# epoch:  3 loss:  2.2592897415161133
# epoch:  4 loss:  2.239185094833374
# epoch:  5 loss:  2.2165775299072266
# epoch:  6 loss:  2.1895644664764404
# epoch:  7 loss:  2.1337430477142334
# epoch:  8 loss:  2.145582437515259
# epoch:  9 loss:  2.07069730758667
# epoch:  10 loss:  2.0160953998565674
# epoch:  11 loss:  2.0097544193267822
# epoch:  12 loss:  1.971535086631775
# epoch:  13 loss:  2.0030770301818848
# epoch:  14 loss:  1.9374500513076782
# epoch:  15 loss:  1.845347285270691
# epoch:  16 loss:  1.8059598207473755
# epoch:  17 loss:  1.8297380208969116
# epoch:  18 loss:  1.7340795993804932
# epoch:  19 loss:  1.7992690801620483

#隐藏层500，输出层28  可以看到，随着隐藏层增加，损失值收敛速度更快
# epoch:  0 loss:  2.3788058757781982
# epoch:  1 loss:  2.3052971363067627
# epoch:  2 loss:  2.2635960578918457
# epoch:  3 loss:  2.2261438369750977
# epoch:  4 loss:  2.188366651535034
# epoch:  5 loss:  2.1712470054626465
# epoch:  6 loss:  2.148013114929199
# epoch:  7 loss:  2.1211185455322266
# epoch:  8 loss:  2.069732904434204
# epoch:  9 loss:  2.0479066371917725
# epoch:  10 loss:  2.0173211097717285
# epoch:  11 loss:  1.9681037664413452
# epoch:  12 loss:  1.9592158794403076
# epoch:  13 loss:  1.8993645906448364
# epoch:  14 loss:  1.849704384803772
# epoch:  15 loss:  1.804060935974121
# epoch:  16 loss:  1.825506567955017
# epoch:  17 loss:  1.760971188545227
# epoch:  18 loss:  1.650210976600647
# epoch:  19 loss:  1.7435760498046875

# 隐藏层 == 输入层，784，输出层 == 28，相同训练次数下，隐藏层神经元越多，损失值越小
# epoch:  0 loss:  2.3356423377990723
# epoch:  1 loss:  2.2844128608703613
# epoch:  2 loss:  2.247927188873291
# epoch:  3 loss:  2.1989858150482178
# epoch:  4 loss:  2.173894166946411
# epoch:  5 loss:  2.1507456302642822
# epoch:  6 loss:  2.1267147064208984
# epoch:  7 loss:  2.0904955863952637
# epoch:  8 loss:  2.032600164413452
# epoch:  9 loss:  1.9734296798706055
# epoch:  10 loss:  1.9697613716125488
# epoch:  11 loss:  1.9230693578720093
# epoch:  12 loss:  1.8963918685913086
# epoch:  13 loss:  1.8147059679031372
# epoch:  14 loss:  1.7912172079086304
# epoch:  15 loss:  1.8045570850372314
# epoch:  16 loss:  1.7493170499801636
# epoch:  17 loss:  1.6672557592391968
# epoch:  18 loss:  1.6895935535430908
# epoch:  19 loss:  1.6713603734970093

# 隐藏层1，输出层 == 28  隐藏层太小，损失值下降梯度小
# epoch:  0 loss:  3.5924105644226074
# epoch:  1 loss:  3.3730945587158203
# epoch:  2 loss:  3.351752996444702
# epoch:  3 loss:  3.1876983642578125
# epoch:  4 loss:  3.165384292602539
# epoch:  5 loss:  3.229018211364746
# epoch:  6 loss:  3.071805000305176
# epoch:  7 loss:  3.1005992889404297
# epoch:  8 loss:  3.024155616760254
# epoch:  9 loss:  3.035543918609619
# epoch:  10 loss:  2.915066719055176
# epoch:  11 loss:  2.9138262271881104
# epoch:  12 loss:  2.840428113937378
# epoch:  13 loss:  2.818676710128784
# epoch:  14 loss:  2.8798067569732666
# epoch:  15 loss:  2.824005365371704
# epoch:  16 loss:  2.7558176517486572
# epoch:  17 loss:  2.721212863922119
# epoch:  18 loss:  2.6336910724639893
# epoch:  19 loss:  2.7024612426757812

#调整批次 batch_size = 500，隐藏层 == 784，输出层 == 28
#较大的 batch_size 可以利用并行计算能力，提高训练速度，但可能会导致内存占用过高，并且收敛可能不够稳定。
# epoch:  0 loss:  2.605450391769409
# epoch:  1 loss:  2.4553959369659424
# epoch:  2 loss:  2.3982958793640137
# epoch:  3 loss:  2.3604795932769775
# epoch:  4 loss:  2.3393948078155518
# epoch:  5 loss:  2.3212499618530273
# epoch:  6 loss:  2.309457540512085
# epoch:  7 loss:  2.2967300415039062
# epoch:  8 loss:  2.2893731594085693
# epoch:  9 loss:  2.2840404510498047
# epoch:  10 loss:  2.2735273838043213
# epoch:  11 loss:  2.265533924102783
# epoch:  12 loss:  2.2607481479644775
# epoch:  13 loss:  2.2531192302703857
# epoch:  14 loss:  2.245236873626709
# epoch:  15 loss:  2.235158920288086
# epoch:  16 loss:  2.2313172817230225
# epoch:  17 loss:  2.223386287689209
# epoch:  18 loss:  2.2176108360290527
# epoch:  19 loss:  2.2128612995147705

# 调整批次 batch_size = 1000
# epoch:  0 loss:  2.81026554107666
# epoch:  1 loss:  2.603799343109131
# epoch:  2 loss:  2.5071449279785156
# epoch:  3 loss:  2.4537715911865234
# epoch:  4 loss:  2.416778802871704
# epoch:  5 loss:  2.3937878608703613
# epoch:  6 loss:  2.3751657009124756
# epoch:  7 loss:  2.3611905574798584
# epoch:  8 loss:  2.34759521484375
# epoch:  9 loss:  2.3386220932006836
# epoch:  10 loss:  2.3319082260131836
# epoch:  11 loss:  2.3240671157836914
# epoch:  12 loss:  2.316589593887329
# epoch:  13 loss:  2.3087635040283203
# epoch:  14 loss:  2.303065538406372
# epoch:  15 loss:  2.2987754344940186
# epoch:  16 loss:  2.2943472862243652
# epoch:  17 loss:  2.2885079383850098
# epoch:  18 loss:  2.2839791774749756
# epoch:  19 loss:  2.280902147293091
# Accuracy: 33.17%

# 调整批次 batch_size = 10  批次数越小，收敛越快，但是训练时间越长
# epoch:  0 loss:  1.9918591976165771
# epoch:  1 loss:  1.597936987876892
# epoch:  2 loss:  1.4618637561798096
# epoch:  3 loss:  1.5967824459075928
# epoch:  4 loss:  0.9014881253242493
# epoch:  5 loss:  1.3423645496368408
# epoch:  6 loss:  0.7842575311660767
# epoch:  7 loss:  0.995740532875061
# epoch:  8 loss:  0.9792119264602661
# epoch:  9 loss:  0.5677952766418457
# epoch:  10 loss:  0.46067729592323303
# epoch:  11 loss:  0.5363357067108154
# epoch:  12 loss:  0.4965365529060364
# epoch:  13 loss:  0.5004640817642212
# epoch:  14 loss:  0.4827187657356262
# epoch:  15 loss:  0.7257678508758545
# epoch:  16 loss:  1.109390139579773
# epoch:  17 loss:  0.1599079966545105
# epoch:  18 loss:  0.13118411600589752
# epoch:  19 loss:  0.25220370292663574
# Accuracy: 67.34%

# 学习率 0.002，epochs = 20 batch_size = 10
# epoch:  0 loss:  1.5606110095977783
# epoch:  1 loss:  0.9399741291999817
# epoch:  2 loss:  0.8812375068664551
# epoch:  3 loss:  0.9242618680000305
# epoch:  4 loss:  0.718248724937439
# epoch:  5 loss:  0.6856020092964172
# epoch:  6 loss:  0.8776515126228333
# epoch:  7 loss:  0.3974875807762146
# epoch:  8 loss:  0.4209781289100647
# epoch:  9 loss:  0.3231658339500427
# epoch:  10 loss:  0.858044445514679
# epoch:  11 loss:  0.6554198265075684
# epoch:  12 loss:  0.9982331991195679
# epoch:  13 loss:  0.20121900737285614
# epoch:  14 loss:  0.5711956024169922
# epoch:  15 loss:  0.526558518409729
# epoch:  16 loss:  0.6589817404747009
# epoch:  17 loss:  0.9541338682174683
# epoch:  18 loss:  1.109783411026001
# epoch:  19 loss:  0.572283148765564
# Accuracy: 68.99%

# learning_rate = 0.002 epochs = 100 batch_size = 100
# epoch:  90 loss:  0.6798588633537292
# epoch:  91 loss:  0.511711061000824
# epoch:  92 loss:  0.512153685092926
# epoch:  93 loss:  0.7753939032554626
# epoch:  94 loss:  0.744831919670105
# epoch:  95 loss:  0.5754738450050354
# epoch:  96 loss:  0.8101463317871094
# epoch:  97 loss:  0.6829956769943237
# epoch:  98 loss:  0.7523707747459412
# epoch:  99 loss:  0.5914349555969238
# Accuracy: 66.60000000000001%

# learning_rate = 0.002 epochs = 100 batch_size = 20
# epoch:  90 loss:  0.3887343406677246
# epoch:  91 loss:  0.5067153573036194
# epoch:  92 loss:  0.39695146679878235
# epoch:  93 loss:  0.6162430644035339
# epoch:  94 loss:  0.33090758323669434
# epoch:  95 loss:  0.4471420347690582
# epoch:  96 loss:  0.4733828008174896
# epoch:  97 loss:  0.15662890672683716
# epoch:  98 loss:  0.34819191694259644
# epoch:  99 loss:  0.4422924518585205
# Accuracy: 73.72999999999999%

# learning_rate = 0.0005 epochs = 100 batch_size = 20
# epoch:  90 loss:  0.8159303665161133
# epoch:  91 loss:  0.86481773853302
# epoch:  92 loss:  0.7598246335983276
# epoch:  93 loss:  1.0103052854537964
# epoch:  94 loss:  0.6713430881500244
# epoch:  95 loss:  0.5879992246627808
# epoch:  96 loss:  0.8252964019775391
# epoch:  97 loss:  1.0627034902572632
# epoch:  98 loss:  0.40611886978149414
# epoch:  99 loss:  0.5443419218063354
# Accuracy: 67.53%