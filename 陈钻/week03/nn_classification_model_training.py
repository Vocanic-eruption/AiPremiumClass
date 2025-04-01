import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


# hyperparameters
learning_rate = 1e-2
batch_size = 64
epochs = 50


training_data = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=ToTensor(), 
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用 {device} 设备")


# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10),
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for data, target in train_dataloader:
        # print(data.shape)
        # print(target.shape)
        pred = model(data.reshape(data.shape[0], -1).to(device))
        loss = loss_fn(pred, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch: {epoch} Loss: {loss.item()}")
    if loss.item() < 0.40:
        break

print(model)
# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')