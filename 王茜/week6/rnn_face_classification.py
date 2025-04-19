from tensorflow.python.ops.gen_linalg_ops import batch_cholesky
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.nn import RNN, LSTM, GRU
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class FaceClassifierDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FaceClassifier(nn.Module):
    def __init__(self, model_name,
                 input_size,
                 hidden_size,
                 num_layer,
                 num_classes):
        super().__init__()
        self.model_name = model_name
        self.num_layers = num_layer
        self.hidden_size = hidden_size
        # rnn layer
        if model_name == 'rnn':
            self.rnn = RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer,
                batch_first=True,
            )
        elif model_name == 'lstm':
            self.rnn = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer,
                batch_first=True,
            )
        elif model_name == 'gru':
            self.rnn = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layer,
                batch_first=True,
            )
        elif model_name == 'bilstm':
            self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            bidirectional=True,  # 关键：设置为双向
            batch_first=True     # 输入维度为 (batch, seq, feature)
        )
        # nn layer
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes,
        )

    def forward(self, X):
        # if self.model_name == 'bilstm':
        #     # 初始化隐藏状态（h0）和细胞状态（c0）
        #     # 双向LSTM需要双倍的隐藏状态：2 * num_layers
        #     # 显示写法
        #     batch_size = X.size(0)
        #     h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        #     c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        #
        #     # 前向传播
        #     output, (hidden, cell) = self.rnn(X, (h0, c0))
        #
        #     # 输出说明：
        #     # out.shape: (batch_size, seq_length, 2*hidden_size)
        #     # hidden.shape: (2*num_layers, batch_size, hidden_size)
        #     # cell.shape: (2*num_layers, batch_size, hidden_size)
        # else:
        #     output, h_n = self.rnn(X)
        # 隐式写法
        output, h_n = self.rnn(X)
        if output.shape[-1] != self.hidden_size:
            y1 = self.classifier(output[..., :self.hidden_size])
            y2 = self.classifier(output[..., self.hidden_size:])
            y = y1 + y2
        else:
            y = self.classifier(output)
        return y


def pred(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for X, labels in dataloader:
            output = model(X)
            _, pred_y = torch.max(output, 1)
            all_outputs.extend(pred_y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_labels) == np.array(all_outputs))
    return accuracy


def load_data(test_size, batch_size):
    X, y = fetch_olivetti_faces(
        data_home='../data',
        return_X_y=True,
        shuffle=True)
    # 对输入数据做归一化（假设数据为图像或连续值）
    X = (X - X.mean()) / X.std()
    # 转换为 PyTorch Tensor
    X_torch = torch.from_numpy(X).float()  # 转换为 float32 类型
    y_torch = torch.from_numpy(y).long()  # 标签建议用 long 类型
    dataset = FaceClassifierDataset(X_torch, y_torch)
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train(writer, model_type, train_loader, test_loader, epochs=100, lr=1e-3):
    # 模型
    model = FaceClassifier(
        model_name=model_type,
        input_size=4096,
        hidden_size=128,
        num_layer=2,
        num_classes=40,
    )
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 纪录每轮训练的损失和准确率
    history_loss = []
    history_train_accuracy = []
    history_test_accuracy = []
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in\
            tqdm(enumerate(train_loader), total=len(train_loader)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch/batch [{epoch}/{batch_idx}],'
                      f' Loss: {loss.item():.4f}')

        train_acc = pred(model, train_loader)
        test_acc = pred(model, test_loader)
        history_loss.append(loss.item())
        history_train_accuracy.append(train_acc)
        history_test_accuracy.append(test_acc)
        writer.add_scalar('Training Accuracy', train_acc, epoch)
        writer.add_scalar('Training Loss (per epoch)', loss.item(), epoch)
    return history_loss, history_train_accuracy, history_test_accuracy


if __name__ == '__main__':
    test_size = 0.2
    batch_size = 10
    epochs = 30
    lr = 1e-3
    model_types = ['rnn', 'lstm', 'gru', 'bilstm']
    # 加载数据集
    train_loader, test_loader = load_data(test_size, batch_size)
    # 纪录训练信息
    loss_df = pd.DataFrame(index=range(epochs), columns=model_types)
    train_acc_df = pd.DataFrame(index=range(epochs), columns=model_types)
    test_acc_df = pd.DataFrame(index=range(epochs), columns=model_types)
    # 创建 SummaryWriter，指定日志保存路径

    # 使用多种模型训练
    for model in model_types:
        writer = SummaryWriter(f'runs/{model}')  # 日志会保存在 runs/experiment_1 目录
        loss, train_acc, test_acc = (
            train(writer, model, train_loader, test_loader, epochs, lr))
        loss_df[model] = loss
        train_acc_df[model] = train_acc
        test_acc_df[model] = test_acc
        writer.close()

    loss_df.plot(label='loss', title='loss')
    plt.show()
    train_acc_df.plot(label='train_acc', title='train_acc')
    plt.show()
    test_acc_df.plot(label='test_acc', title='test_acc')
    plt.show()