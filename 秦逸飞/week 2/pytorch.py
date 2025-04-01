import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.v2 import ToTensor     
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader 
LR = 1e-3
epochs = 20
BATCH_SIZE = 128
train_data = FashionMNIST(root='./fashion_data', train=True, download=True, 
                          transform=ToTensor())
test_data = FashionMNIST(root='./fashion_data', train=False, download=True,
                         transform=ToTensor())
trian_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.Sigmoid(),
    nn.Linear(64, 10)
)
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
for epoch in range(epochs):
    for data, target in trian_dl:
        output = model(data.reshape(-1, 784))
        loss = loss_fn(output, target)
        optimizer.zero_grad()  
        loss.backward()    
        optimizer.step()   

    print(f'Epoch:{epoch} Loss: {loss.item()}')
  test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)

correct = 0
total = 0
with torch.no_grad():  
    for data, target in test_dl:
        output = model(data.reshape(-1, 784))
        _, predicted = torch.max(output, 1)  
        total += target.size(0)  # size(0) 等效 shape[0]
        correct += (predicted == target).sum().item()

print(f'Accuracy: {correct/total*100}%')






import torch
from torchvision.transforms.v2 import ToTensor    
from torchvision.datasets import FashionMNIST
train_data = FashionMNIST(root='./fashion_data', train=True, download=True, 
                          transform=ToTensor())
test_data = FashionMNIST(root='./fashion_data', train=False, download=True,
                         transform=ToTensor())
train_data
train_data[1]
train_data[1][0].shape 
train_data[1][0].reshape(-1).shape
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./fashion_data', train=True, download=True)

img,clzz = train_data[12301]
plt.imshow(img, cmap='gray') 
plt.title(clzz)
plt.show()
