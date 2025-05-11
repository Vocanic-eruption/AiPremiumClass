from sklearn.datasets import fetch_olivetti_faces
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNN_Classifier(torch.nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type
        # 普通循环神经网络
        if (type == 'rnn'):
            self.rnn = torch.nn.RNN(
                input_size=64, # 输入特征数
                hidden_size=300, # 隐藏层特征数
                num_layers=3, # 隐藏层层数
                bias=True,
                batch_first=True, # 输入输出的数据格式为 (batch, seq, feature)
            )
        # 门控循环神经网络
        elif (type == 'gru'):
            self.rnn = torch.nn.GRU(
                input_size=64,
                hidden_size=300,
                num_layers=3,
                bias=True,
                batch_first=True,
            )
        # 长短期记忆循环神经网络
        elif (type == 'lstm'):
            self.rnn = torch.nn.LSTM(
                input_size=64,
                hidden_size=300,
                num_layers=3,
                bias=True,
                batch_first=True,
            )
        # 线性神经网络
        self.linear = torch.nn.Linear(300, 40)
    def forward(self, x):
        if (self.type == 'lstm'):
            _, (lh, _) = self.rnn(x) # 返回值：output, (h_n, c_n)，其中c_n表示最后一个时间步的所有层的细胞状态，LSTM 特有
        else:
            _, lh = self.rnn(x) # lh 形状为 (num_layers, batch_size, hidden_size)
        # out = self.linear(lh.reshape(-1, 128)) 
        out = self.linear(lh[-1]) # 注意：只取最后一层的隐藏状态，形状为 (batch_size, hidden_size)，上面reshape的方法在num_layers>1时会有问题
        return out

def train_test(model):
    writer = SummaryWriter(log_dir=f'./runs/{model.type}')
    # 交叉熵损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        model.train()
        for i, (img, lab) in enumerate(train_dl):
            out = model(img)
            loss = loss_fn(out, lab)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) # 梯度裁剪
            optimizer.step()
            writer.add_scalar('train loss', loss.item(), len(train_dl) * epoch + i)
        print(f'epoch={epoch} ,loss={loss.item()}')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for img, lab in test_dl:
                out = model(img)
                pred = torch.argmax(out, dim=1)
                correct += (pred == lab).sum().item()
                total += lab.size(0)
            acc = 100*correct / total
            writer.add_scalar('test acc', acc, epoch)
            print(f'epoch={epoch} ,acc={acc}')
    writer.close()

if __name__ == '__main__':
    # 加载人脸数据
    olivetti_faces = fetch_olivetti_faces(data_home='./face_data')
    # 其中：olivetti_faces.images.shape = (400, 64, 64)
    datasets = [(img, lab) for img, lab in zip(torch.tensor(olivetti_faces.images), torch.tensor(olivetti_faces.target))]
    # 划分训练集和测试集
    train_data, test_data = train_test_split(datasets, test_size=0.2)
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=32, shuffle=False)

    model_rnn = RNN_Classifier('rnn')
    train_test(model_rnn)
    model_gru = RNN_Classifier('gru')
    train_test(model_gru)
    model_lstm = RNN_Classifier('lstm')
    train_test(model_lstm)