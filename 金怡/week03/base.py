# import torch
# from torchvision.datasets import FashionMNIST
#
# train_data=FashionMNIST(root='/fashion_data',train=True,download=True)
# test_data=FashionMNIST(root='/fashion_data',train=True,download=True)
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
# img,clzz=train_data[0]
# plt.imshow(img,cmap='gray')
# plt.title(clzz)
# plt.show()
# crossentropyloss?
# loss_fn=nn.CrossEntropyLoss()
# optimizer=torch.optim.SGD(mdoel.parameters(),lr=LR)
# for epoch in epochs:
#     for data,target in train_data:
#         output=model(data)#forward
#         # -> output=model(data.reshape(Batch_szie,-1))否则一个个样本计算效率很低
#         los=loss_fn(output,target)
#         optimizer.zero_grad()#clear gradient
#         loss.backward()# calculate gradient
#         optimizer.step()#update coefficient
#     model()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader

LR = 0.01
BATCH_SIZE = 64
EPOCHS = 10

# pre processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = KMNIST(root='./data', train=True, transform=transform, download=True)
test_data = KMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = NeuralNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)


def train(model, train_loader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            output = model(data)  # forward
            loss = loss_fn(output, target)
            optimizer.zero_grad()  # clear gradient
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


train(model, train_loader, loss_fn, optimizer, EPOCHS)
test(model, test_loader)
