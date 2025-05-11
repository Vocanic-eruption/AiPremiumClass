# encoding: utf-8
# @File  : RNNClassifier.py
# @Author: GUIFEI
# @Desc : 使用Rnn 实现 olivettiFaces 数据集人脸分类
# @Date  :  2025/04/08
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNNClassifier(nn.Module):
    def __init__(self, model_name="LSTM"):
        super(RNNClassifier, self).__init__()
        self.rnn = None
        self.init_rnn(model_name)
        self.fc = nn.Linear(128, 40)  # 输出层

    def init_rnn(self, model_name):
        if model_name == "LSTM":
            self.rnn = nn.LSTM(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True
            )
        elif model_name == "GRU":
            self.rnn = nn.GRU(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True
            )
        elif model_name == "RNN":
            self.rnn = nn.RNN(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True
            )
        elif model_name == "BiLSTM":
            self.rnn = nn.LSTM(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True,
                bidirectional=True
            )
        elif model_name == "BiRNN":
            self.rnn = nn.RNN(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True,
                bidirectional=True
            )
        elif model_name == "BiGRU":
            self.rnn = nn.GRU(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bias=True,
                bidirectional=True
            )
        else:
            raise ValueError("Unsupported RNN type.")


    def forward(self, x):
        '''
        定义前向运算
        :param x:
        :return:
        '''
        outputs, h_l = self.rnn(x)
        out = self.fc(outputs[:,-1,:])
        return out

def load_data():
    # 加载数据
    olivetti_faces = fetch_olivetti_faces(data_home="../dataset/olivettiFaces")
    # 此处注意取 images 而非 data
    data, target = olivetti_faces.images, olivetti_faces.target
    # 划分训练集及测试集，并设置随机数种子，使用随机数种子确保代码每次运行时数据一致
    # stratify=olivetti_faces.target 按照target 维度对数据集进行分层采样，却表40个人的数据都会被采集到
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42,
                                                                        stratify=target)
    # 将特征数据与目标值结合在一起
    train_data_set = TensorDataset(torch.tensor(data_train, dtype=torch.float),
                                   torch.tensor(target_train, dtype=torch.long))
    test_data_set = TensorDataset(torch.tensor(data_test, dtype=torch.float),
                                  torch.tensor(target_test, dtype=torch.long))
    train_dataloader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

# 定义优化器及损失函数
def build_rnn_model(model_name):
    loss_fn = nn.CrossEntropyLoss()
    model = RNNClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return loss_fn, model, optimizer


if __name__ == '__main__':
    # 定义超参数
    EPOCHS = 200
    BATCH_SIZE = 32
    LR = 0.001
    # 加载数据
    train_dataloader, test_dataloader = load_data()
    # 初始化 tensorboard
    writer = SummaryWriter()
    # 初始化模型
    model_name_list = ['LSTM', 'GRU', 'RNN', 'BiLSTM', 'BiRNN', 'BiGRU']
    for model_name in model_name_list:
        loss_fn, model, optimizer = build_rnn_model(model_name)
        for epoch in range(EPOCHS):
            model.train()
            for i, (images, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                # 去除矩阵中值为1 的维度（颜色通道）
                outputs = model(images.squeeze())
                loss = loss_fn(outputs, labels)
                loss.backward()
                # 使用梯度裁剪，并设置最大L2范数为1.0， 超过这个值则会进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{EPOCHS}], {model_name} Loss: {loss.item():.4f}')
                    writer.add_scalar(f'{model_name} training loss', loss.item(), epoch * len(train_dataloader) + i)
            # 评估模型
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_dataloader:
                    outputs = model(images.squeeze())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch + 1}/{EPOCHS}], {model_name} Test Accuracy: {accuracy:.2f}%')
                writer.add_scalar(f'{model_name} test accuracy', accuracy, epoch)





