import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNN_Classifier(nn.Module):
    def __init__(self):
        super(RNN_Classifier, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=256, bias=True, num_layers=3, batch_first=True)
        self.fc = nn.Linear(256, 40)

    def forward(self, x):
        outputs, l_h = self.rnn(x)
        outlast = self.fc(outputs[:, -1, :])
        return outlast


if __name__ == '__main__':
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # olivetti_faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
    # images = torch.tensor(olivetti_faces.data)  # images.shape = (400, 64, 64)
    # targets = torch.tensor(olivetti_faces.target)  # targets.shape = (400,)
    images = torch.from_numpy(np.load('./face_data/olivetti_faces.npy'))
    targets = torch.from_numpy(np.load('./face_data/olivetti_faces_target.npy')).long()
    dataset = [(img, lbl) for img, lbl in zip(images, targets)]
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    model = RNN_Classifier()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.squeeze())
            loss =criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            writer.add_scalar('train loss', loss, epoch * len(train_loader) + i)  # ✅ 唯一全局步数


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
            print(f'Epoch [{epoch+1} / {num_epochs}], test accuracy:{accuracy:.2f}%')
            writer.add_scalar('test accuracy', accuracy, epoch)
            writer.add_graph(model, images)  # 可视化网络结构


    torch.save(model, 'rnn_olivefaces.pth')
    torch.save(model.state_dict(), 'rnn_olivefaces_params.pth')
    writer.close()



    # model = torch.load('rnn_model.pth')
    # model = RNN_Classifier()
    # model.load_state_dict(torch.load('rnn_model_params.pth'))


'''
运行结果：
D:\soft\anacona-tool\envs\py12\python.exe D:/AI-money-code/old_NLP/badou-week06/rnn_olivetti_faces.py
Epoch [1/50], Loss: 3.6871
Epoch [1/50], Loss: 3.7686
Epoch [1/50], Loss: 3.6763
Epoch [1/50], Loss: 3.6737
Epoch [1/50], Loss: 3.7013
Epoch [1/50], Loss: 3.6984
Epoch [1/50], Loss: 3.6718
Epoch [1/50], Loss: 3.7253
Epoch [1/50], Loss: 3.7178
Epoch [1/50], Loss: 3.6665
Epoch [1 / 50], test accuracy:6.25%
Epoch [2/50], Loss: 3.4953
Epoch [2/50], Loss: 3.6001
Epoch [2/50], Loss: 3.5236
Epoch [2/50], Loss: 3.4518
Epoch [2/50], Loss: 3.4226
Epoch [2/50], Loss: 3.5703
Epoch [2/50], Loss: 3.5073
Epoch [2/50], Loss: 3.4046
Epoch [2/50], Loss: 3.4043
Epoch [2/50], Loss: 3.5354
Epoch [2 / 50], test accuracy:7.50%
Epoch [3/50], Loss: 3.2847
Epoch [3/50], Loss: 3.2530
Epoch [3/50], Loss: 3.5132
Epoch [3/50], Loss: 3.2719
Epoch [3/50], Loss: 3.1611
Epoch [3/50], Loss: 3.2679
Epoch [3/50], Loss: 3.1689
Epoch [3/50], Loss: 3.1111
Epoch [3/50], Loss: 3.0323
Epoch [3/50], Loss: 3.1200
Epoch [3 / 50], test accuracy:12.50%
........
Epoch [47 / 50], test accuracy:66.25%
Epoch [48/50], Loss: 0.3522
Epoch [48/50], Loss: 0.2911
Epoch [48/50], Loss: 0.3016
Epoch [48/50], Loss: 0.4256
Epoch [48/50], Loss: 0.2477
Epoch [48/50], Loss: 0.4411
Epoch [48/50], Loss: 0.4388
Epoch [48/50], Loss: 0.6212
Epoch [48/50], Loss: 0.4773
Epoch [48/50], Loss: 0.7321
Epoch [48 / 50], test accuracy:65.00%
Epoch [49/50], Loss: 0.2360
Epoch [49/50], Loss: 0.2909
Epoch [49/50], Loss: 0.3740
Epoch [49/50], Loss: 0.7121
Epoch [49/50], Loss: 0.4464
Epoch [49/50], Loss: 0.4332
Epoch [49/50], Loss: 0.6180
Epoch [49/50], Loss: 0.4262
Epoch [49/50], Loss: 0.3257
Epoch [49/50], Loss: 0.5294
Epoch [49 / 50], test accuracy:61.25%
Epoch [50/50], Loss: 0.4492
Epoch [50/50], Loss: 0.3221
Epoch [50/50], Loss: 0.5461
Epoch [50/50], Loss: 0.4139
Epoch [50/50], Loss: 0.2687
Epoch [50/50], Loss: 0.1991
Epoch [50/50], Loss: 0.3090
Epoch [50/50], Loss: 0.2126
Epoch [50/50], Loss: 0.2135
Epoch [50/50], Loss: 0.3409
Epoch [50 / 50], test accuracy:61.25%

进程已结束，退出代码为 0

'''