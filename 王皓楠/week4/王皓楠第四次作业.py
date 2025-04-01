# %%
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn as nn

# %%
train_data=fetch_olivetti_faces(data_home='./face_data',shuffle=True)
image=torch.tensor(train_data.data)
targets=torch.tensor(train_data.target)

# %% [markdown]
# olvetti_face数据集手动进行分割，方法是首先生成一个indices序列包含所有下标，然后利用stratify将标签label传入进去保证数据分布比例不变，在利用test_size进行分割。

# %%
datasets=[(img,lbl)for img,lbl in zip(image,targets)]
labels = torch.stack([d[1] for d in datasets])  # (N,)
#需要手动分割测试集和验证集
#生成分层索引（按标签分层）
indices=np.arange(len(datasets))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,         # 20%验证集
    stratify=labels.numpy(),# 保持类别分布
    random_state=42        # 固定种子
)
# 2. 分割数据集列表
train_ds = [datasets[i] for i in train_idx]
val_ds = [datasets[i] for i in val_idx]

#小批量分割
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=True)

from torch_myn import Torch_nn
print(datasets[0][1].shape)
train_ds

# %%
LR=1e-3
epochs=100

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# 模型定义,定义三个模型对应三个优化器

# %%
#把模型直接放到GPU上
model1=Torch_nn()
model2=Torch_nn()
model3=Torch_nn()
model1=model1.to(device)
model2=model2.to(device)
model3=model3.to(device)

models=[model1,model2,model3]
#损失计算
loss_fn=nn.CrossEntropyLoss()
#设置三个不同优化器
optmizer1=torch.optim.SGD(model1.parameters(),momentum=0.9,lr=LR)
optmizer2=torch.optim.RMSprop(model2.parameters(),alpha=0.99,lr=LR)
optmizer3=torch.optim.AdamW(model3.parameters(),weight_decay=0.01,lr=LR)
optimizers=[optmizer1,optmizer2,optmizer3]

# %%
!nvidia-smi

# %%
#对比看不同模型的Loss输出
for idx,(model,optimizer) in enumerate(zip(models,optimizers)):
    #使得生效
    model.train()
    for epoch in range(epochs): 
        for X,y in dataloader:
            #把数据转型后放到GPU上
            y=y.long()
            X,y=X.to(device),y.to(device)
            #前向传播
            output=model(X.reshape(-1,4096))
            #损失函数计算
            loss=loss_fn(output,y)
            #梯度清零
            model.zero_grad()
            #梯度计算
            loss.backward()
            #反向传播
            optimizer.step()
        print(f'Model:{idx+1},Epoch:{epoch},Loss:{loss.item()}')

# %% [markdown]
# 测试时候测试3个不同模型，测试时发现优化器RMSprop损失函数波动很大，SGD稳步下降，波动比较小，而AdamW损失函数小幅波动

# %%
#测试时候直接拿出来
test_dl=torch.utils.data.DataLoader(val_ds, shuffle=True)
for idx,model in enumerate(models):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        for X,y in test_dl:
            y=y.long()
            X, y = X.to(device), y.to(device)
            
            output = model(X.reshape(-1, 4096))
            val_loss += loss_fn(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
    
    # 打印结果
    print(f"Model:{idx+1},Epoch {epoch+1} | Val Loss: {val_loss/len(test_dl):.4f} | Val Acc: {correct/len(val_ds):.4f}")


