import torch
import torch.nn as nn
import os
import shutil
from matplotlib.pylab import f
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):
    def __init__(self, name, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        if name == 'Normal-RNN':
            nn_name = nn.RNN
        elif name == 'LSTM':
            nn_name = nn.LSTM
        elif name == 'GRU':
            nn_name = nn.GRU

        self.rnn = nn_name(
            input_size=input_size,   # x的特征维度
            hidden_size=hidden_size,  # 隐藏层神经元数量
            bias=True,        # 偏置
            num_layers=num_layers,     # 隐藏层层数
            batch_first=True  # 批次是输入第一个维度
        )
        self.classifier = nn.Linear(hidden_size, num_classes)  # 输出层

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.classifier(outputs[:,-1,:])
        return out

# 定义自定义数据集类
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 将数据转换为Tensor格式
        image = torch.tensor(self.X[idx], dtype=torch.float).view(1, 64, 64)  # 64x64图像
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)  # 确保标签是整数
        return image, label

if __name__ == '__main__':

    datasets_path = "../../../../datasets_data/faces_data"

    log_dir = './logs/homework1_RNN'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Deleted existing {log_dir} directory.")

    writer = SummaryWriter(log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    olivetti_faces = fetch_openml(name='olivetti_faces',data_home=datasets_path,
                                  version=1, as_frame=False)

    X, y = olivetti_faces.data, olivetti_faces.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 输出数据集的形状以验证划分
    # print(f"训练集特征形状: {X_train.shape}") # (320, 4096)
    # print(f"测试集特征形状: {X_test.shape}")  # (80, 4096)
    # print(f"训练集目标形状: {y_train.shape}") # (320,)
    # print(f"测试集目标形状: {y_test.shape}") # (80,)
    # print(f"数据集标签数量: {len(set(y))}") # 40

    # 使用自定义数据集类实例化训练集和测试集
    train_dataset = FaceDataset(X_train, y_train)
    test_dataset = FaceDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 打印数据加载器输出的形状
    # for images, labels in train_loader:
    #     print('images shape:', images.shape) # torch.Size([64, 1, 28, 28])
    #     print('labels shape:', labels.shape) # torch.Size([64])
    #     break

    # 参数设置
    input_size=64
    hidden_size=100
    num_layers=2
    num_classes=40

    # 实例化模型
    models = {
        'Normal-RNN': RNN_Classifier('Normal-RNN', input_size, hidden_size, num_layers, num_classes),
        'LSTM': RNN_Classifier('LSTM', input_size, hidden_size, num_layers, num_classes),
        'GRU': RNN_Classifier('GRU', input_size, hidden_size, num_layers, num_classes)
    }

    model_names = list(models.keys())
    model_arr = list(models.values())

    optimizer_list = {}

    for name, model in zip(model_names, model_arr):
        # print(f"Model Name: {name}, Model Instance: {model}")
        model.to(device)
        optimizer_list[name] = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 100
    for j, (name, model) in enumerate(models.items()):
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                # print(f"{images.squeeze().shape}")
                images, labels = images.to(device), labels.to(device)
                optimizer_list[name].zero_grad()
                outputs = model(images.squeeze())
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer_list[name].step()

                if i % 100 == 0:
                    print(f'{name}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                    writer.add_scalars('training loss',
                                       {model_names[j]: loss.item()}, epoch * len(train_loader) + i)
            # 评估模型
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.squeeze())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'{name}: Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')
                writer.add_scalar('test accuracy', {model_names[j]: accuracy}, epoch)