from sklearn.datasets import fetch_olivetti_faces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 数据集加载
olivetti_faces = fetch_olivetti_faces(data_home='./data', shuffle=False, download_if_missing=True)

images = torch.tensor(olivetti_faces.data, dtype=torch.float32)
target = torch.tensor(olivetti_faces.target, dtype=torch.long)
dataset = [(img, lbl) for img, lbl in zip(images, target)]

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# 模型构建
class FaceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, rnn_type='RNN', bidirectional=False):
        super().__init__()
        # 根据rnn_type选择不同的RNN结构
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("Unsupported RNN type. Choose from ['RNN', 'LSTM', 'GRU']")

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(hidden_size * (2 if bidirectional else 1), num_labels)

    def forward(self, input_data):
        if isinstance(self.rnn, nn.LSTM):
            output, (h_n, _) = self.rnn(input_data)
        else:
            output, h_n = self.rnn(input_data)

        if self.bidirectional:
            h_n = h_n.view(2, -1, self.hidden_size).sum(dim=0)  # 合并双向的 hidden state

        return self.classifier(h_n.squeeze(0))

# 定义超参数
EPOCHS = 60
HIDDEN_SIZE = 128
INPUT_SIZE = 4096
LEARNING_RATE = 1e-3

# 创建TensorBoard记录器
writer = SummaryWriter(log_dir='./钟文迪/week06_循环神经网络/runs/face_classification')

# 定义训练函数
def train_model(rnn_type, bidirectional=False):
    num_labels = target.max().item() + 1
    model = FaceModel(INPUT_SIZE, HIDDEN_SIZE, num_labels, rnn_type=rnn_type, bidirectional=bidirectional)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (img, lbl) in enumerate(tqdm(dataloader)):
            logits = model(img.reshape(img.size(0), 1, -1))
            loss = loss_fn(logits, lbl)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            model.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == lbl).sum().item()
            total += lbl.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        writer.add_scalar(f'{rnn_type}_Loss', epoch_loss, epoch)
        writer.add_scalar(f'{rnn_type}_Accuracy', epoch_acc, epoch)

    return model

print("Training RNN...")
train_model('RNN')

print("Training LSTM...")
train_model('LSTM')

print("Training GRU...")
train_model('GRU')

# 关闭 TensorBoard
writer.close()