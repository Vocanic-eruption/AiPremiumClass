#pytorch框架处理流程
#1、获取加载数据dataset，然后利用dataloader将数据分成一个batch一个batch(打包快递)，目的是提高训练速度；dataloader用法:dataloader(dataset,batch_size,shuffle=status)
#2、定义模型结构,大部分模型都是继承自nn.Module：
class NeuralNetwork(nn.Module):
    def __init__(self):                  #定义模型网络结构必须要有的就是__init__和forward,
        super().__init__()               #super().__init__()表示调用父类构造函数，init就是构造、初始化
        self.flatten = nn.Flatten()      #然后可以更具功能编写要使用的函数，比如当前分类任务的图像数据是28*28就使用Flatten()直接拉成一维度(==reshape(-1))
        self.kpipeline = nn.Sequential(  # nn.Sequetial表示将一些操作打包起来一起执行，更便利
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)          #最后输出的是这个x分别对这任务10个类别的概率是多少，概率最大的就是模型预测的
        )
    def forward(self,x):              #实例化模型时会自动调用init，然后传入参数调用model时的参数就是自动到forward中，计算返回的值就是预测得到的10个类别不同的概率
        x = self.flatten(x)
        logits = self.kpipeline(x)
        return logits
model = NeuralNetwork()
print(model)
#3、定义模型损失函数和优化器
loss_fn = nn.CrossEntropyLoss()                           #分类任务，回归任务通常使用MSE()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)  #模型优化器就是指定模型所有参数按照(步长 下山)学习率
#4、训练函数
def train(dataloader,model,loss,optimizer):
    model.train()                                         #将模型设置为训练模式
    for batch,(x,y) in enumerate(dataloader):             #这里传进来的训练数据train_dataloader利用eunmerate特性分开，dataloader打包后给解包为样本x，标签y
        x, y = x.to(device),y.to(device)                  #x:64，1，28，28 ,y:64(10个类别)
        pred= model(x)                                    #训练过程就是先将tensor数据拿到转到当前设备上cpu，然后将训练数据传入model，这里模型实际上就是拿到
        loss = loss_fn(pred, y)                           #训练数据直接放在模型的forward中，向前传播，forward向前传播后的值return出来就是pred
        model.zero_grad()                                 #得到预测的值然后和真实值y一起放到损失中，得到损失loss
        loss.backward()                                   #然后模型需要将梯度清空， 然后损失loss反向传播，然后优化器optimizer.step更新参数
        optimizer.step()
        if batch % 100 == 0:
            print(f'batch:{batch},loss:{loss:.6f}')
#5、训练查看结果 增加两层中间层和使用Adam优化器
epochs = 5
for i in range(epochs):
    print(f'当前epoch:{i}')
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print(f'训练结束')
