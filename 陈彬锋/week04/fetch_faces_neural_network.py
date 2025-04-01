import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

faces = fetch_olivetti_faces(data_home="./face",shuffle=True)
print(faces.data.shape)

images = faces.data
targets = faces.target

dataset = [(img,lbl) for img,lbl in zip(images,targets)]

data_loader = DataLoader(dataset,batch_size=10,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = nn.Sequential(
    nn.Linear(4096,2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(2048,2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(2048,1024),
    nn.BatchNorm1d(1024),# 输出通道归一化
    nn.ReLU(),
    nn.Dropout(0.75),# 随机丢弃75%的神经元连接
    nn.Linear(1024,40)
).to(device)

loss_fn = nn.CrossEntropyLoss()
# optimer = optim.SGD(params=model.parameters(),lr=1e-3)
# optimer = optim.SGD(params=model.parameters(),lr=1e-3)
# optimer = optim.Adam(params=model.parameters(),lr=1e-3)
# optimer = optim.AdamW(params=model.parameters(),lr=1e-3)
optimer = optim.RMSprop(params=model.parameters(),lr=1e-3)

loss_hist =[]
for i in range(10):
    for img,lbl in data_loader:
        img,lbl = img.to(device),lbl.to(device)
        result = model(img)
        loss = loss_fn(result,lbl)
        loss.backward()
        optimer.step()
        optimer.zero_grad()
        loss_hist.append(loss.item())
        print(f"loss:{loss.item():.4f}")
