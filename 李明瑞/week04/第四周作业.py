# %%
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

# %%
olivetti_faces = fetch_olivetti_faces(data_home='.', shuffle=True)
print(olivetti_faces.data.shape)
print(olivetti_faces.target.shape)
print(olivetti_faces.images.shape)


# %%
import matplotlib.pyplot as plt
face = olivetti_faces.images[3]
plt.imshow(face, cmap='gray')
plt.show()

# %%
olivetti_faces.data[0]

# %%
olivetti_faces.target

# %%
import torch
import torch.nn as nn

# %%
images = torch.tensor(olivetti_faces.data)
targets = torch.tensor(olivetti_faces.target)

# %%
images.shape

# %%
targets.shape

# %%
dataset = [(img,lbl) for img, lbl in zip(images, targets)]
dataset[0]

# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# %%
device = 'cpu'

# %%
model = nn.Sequential(
    nn.Linear(4096, 8192),
    nn.BatchNorm1d(8192),
    nn.ReLU(),
    nn.Linear(8192, 16384),
    nn.BatchNorm1d(16384),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(16384, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 40)
).to(device)

# %%
print(model)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
lost_hist = []
for i in range(10):
    for img,lbl in dataloader:
        img,lbl = img.to(device),lbl.to(device).long()
        result = model(img)
        loss = criterion(result,lbl)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lost_hist.append(loss.item())
        print(f" loss:{loss.item():.4f}")



