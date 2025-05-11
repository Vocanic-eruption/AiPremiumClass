from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(
            input_size = 64,
            hidden_size = 50,
            bias = True,
            batch_first = True,
            num_layers = 2
        )
        self.fc = nn.Linear(50, 40)
    def forward(self, x):
        outputs, _ = self.rnn(x)
        out = self.fc(outputs[:, -1, :])
        return out



if __name__ == '__main__':

    data = fetch_olivetti_faces(data_home="E:\\study\\AI\\week4_pytorch模型训练相关要素\\data\\faces"
                                , download_if_missing=True)
    images = torch.tensor(data.images, dtype=torch.float32)
    labels = torch.tensor(data.target, dtype=torch.long)
    images = images.view(-1, 64, 64)  # (batch_size, sq_len, input_size)

    dataset = torch.utils.data.TensorDataset(images, labels)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRU().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    writer = SummaryWriter(log_dir='./runs/GRU')
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0,0,0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            loop.set_postfix(loss = loss.item())
        accuracy = correct / total * 100
        total_loss /= len(train_loader)
        writer.add_scalar('Loss_train', total_loss, epoch)
        writer.add_scalar('Accuracy_train', accuracy, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
        model.eval()
        with torch.no_grad():
            val_loss, correct, total = 0,0,0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = correct / total *100
            val_loss /= len(val_loader)
            writer.add_scalar('Loss_val', val_loss, epoch)
            writer.add_scalar('Accuracy_val', accuracy, epoch)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        writer.flush()
    writer.close()

