import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class GRU_Classifier(nn.Module):
    def __init__(self):
        super(GRU_Classifier, self).__init__()
        self.gru = nn.GRU(input_size=64, hidden_size=256, bias=True,
                          num_layers=2, batch_first=True,
                          dropout=0.3, bidirectional=False)
        self.fc = nn.Linear(256, 40)

    def forward(self, x):
        outputs, l_h = self.gru(x)
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


    model = GRU_Classifier()
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


    torch.save(model, 'gru_olivefaces.pth')
    torch.save(model.state_dict(), 'gru_olivefaces_params.pth')
    writer.close()



    # model = torch.load('gru_model.pth')
    # model = GRU_Classifier()
    # model.load_state_dict(torch.load('gru_model_params.pth'))


'''
运行结果
Epoch [1/50], Loss: 3.7182
Epoch [1/50], Loss: 3.7156
Epoch [1/50], Loss: 3.6965
Epoch [1/50], Loss: 3.6214
Epoch [1/50], Loss: 3.7830
Epoch [1/50], Loss: 3.7744
Epoch [1/50], Loss: 3.6016
Epoch [1/50], Loss: 3.6728
Epoch [1/50], Loss: 3.7661
Epoch [1/50], Loss: 3.7761
Epoch [1/50], Loss: 3.7358
Epoch [1/50], Loss: 3.7414
Epoch [1/50], Loss: 3.6792
Epoch [1/50], Loss: 3.7281
Epoch [1/50], Loss: 3.6993
Epoch [1/50], Loss: 3.6904
Epoch [1/50], Loss: 3.6939
Epoch [1/50], Loss: 3.6097
Epoch [1/50], Loss: 3.6328
Epoch [1/50], Loss: 3.6265
Epoch [1 / 50], test accuracy:3.75%
Epoch [2/50], Loss: 3.5747
Epoch [2/50], Loss: 3.6327
Epoch [2/50], Loss: 3.6233
Epoch [2/50], Loss: 3.6910
Epoch [2/50], Loss: 3.5686
Epoch [2/50], Loss: 3.5309
Epoch [2/50], Loss: 3.5198
Epoch [2/50], Loss: 3.5231
Epoch [2/50], Loss: 3.4993
Epoch [2/50], Loss: 3.5888
Epoch [2/50], Loss: 3.4952
Epoch [2/50], Loss: 3.4718
.......
Epoch [49 / 50], test accuracy:77.50%
Epoch [50/50], Loss: 0.0035
Epoch [50/50], Loss: 0.0051
Epoch [50/50], Loss: 0.0054
Epoch [50/50], Loss: 0.0055
Epoch [50/50], Loss: 0.0033
Epoch [50/50], Loss: 0.0081
Epoch [50/50], Loss: 0.0041
Epoch [50/50], Loss: 0.0037
Epoch [50/50], Loss: 0.0047
Epoch [50/50], Loss: 0.0024
Epoch [50/50], Loss: 0.0029
Epoch [50/50], Loss: 0.0028
Epoch [50/50], Loss: 0.0028
Epoch [50/50], Loss: 0.0043
Epoch [50/50], Loss: 0.0046
Epoch [50/50], Loss: 0.0040
Epoch [50/50], Loss: 0.0038
Epoch [50/50], Loss: 0.0026
Epoch [50/50], Loss: 0.0054
Epoch [50/50], Loss: 0.0031
Epoch [50 / 50], test accuracy:82.50%
进程已结束，退出代码为 0
'''
