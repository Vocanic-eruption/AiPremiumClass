import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class LSTM_Classifier(nn.Module):
    def __init__(self):
        super(LSTM_Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=256,
                            bias=True, num_layers=3, batch_first=True,
                            dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(256 * 2, 40)

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
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


    model =LSTM_Classifier()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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


    torch.save(model, 'lstm_olivefaces.pth')
    torch.save(model.state_dict(), 'lstm_olivefaces_params.pth')
    writer.close()



    # model = torch.load('lstm_model.pth')
    # model = LSTM_Classifier()
    # model.load_state_dict(torch.load('lstm_model_params.pth'))


'''运行结果
Epoch [1/50], Loss: 3.7117
Epoch [1/50], Loss: 3.6901
Epoch [1/50], Loss: 3.6997
Epoch [1/50], Loss: 3.7195
Epoch [1/50], Loss: 3.8469
Epoch [1/50], Loss: 3.7183
Epoch [1/50], Loss: 3.7019
Epoch [1/50], Loss: 3.6505
Epoch [1/50], Loss: 3.6437
Epoch [1/50], Loss: 3.8618
Epoch [1/50], Loss: 3.7369
Epoch [1/50], Loss: 3.7605
Epoch [1/50], Loss: 3.7571
Epoch [1/50], Loss: 3.6698
Epoch [1/50], Loss: 3.6410
Epoch [1/50], Loss: 3.6535
Epoch [1/50], Loss: 3.7648
Epoch [1/50], Loss: 3.7511
Epoch [1/50], Loss: 3.6594
Epoch [1/50], Loss: 3.9163
Epoch [1 / 50], test accuracy:1.25%
.........
Epoch [49 / 50], test accuracy:58.75%
Epoch [50/50], Loss: 0.1149
Epoch [50/50], Loss: 0.5601
Epoch [50/50], Loss: 0.4200
Epoch [50/50], Loss: 0.1053
Epoch [50/50], Loss: 0.3355
Epoch [50/50], Loss: 0.6182
Epoch [50/50], Loss: 0.2820
Epoch [50/50], Loss: 0.0805
Epoch [50/50], Loss: 0.0831
Epoch [50/50], Loss: 0.1522
Epoch [50/50], Loss: 0.4723
Epoch [50/50], Loss: 0.0697
Epoch [50/50], Loss: 0.0442
Epoch [50/50], Loss: 0.1075
Epoch [50/50], Loss: 0.3798
Epoch [50/50], Loss: 0.0974
Epoch [50/50], Loss: 0.3237
Epoch [50/50], Loss: 0.1863
Epoch [50/50], Loss: 0.1298
Epoch [50/50], Loss: 0.2842
Epoch [50 / 50], test accuracy:58.75%
'''
