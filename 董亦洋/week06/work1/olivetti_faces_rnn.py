import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class RNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128,40)
    def forward(self,input):
        output,l_h=self.rnn(input)
        out = self.fc(output[:,-1,:])
        return out
    
class GRU_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128,40)
    def forward(self,input):
        output,l_h=self.gru(input)
        out = self.fc(output[:,-1,:])
        return out
    
class LSTM_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128,40)
    def forward(self,input):
        output,l_h=self.lstm(input)
        out = self.fc(output[:,-1,:])
        return out
    
#训练模型
def train(model,criterion,optimizer, train_data, train_target, test_data, test_target):
    writer = SummaryWriter()
    EPOCHS = 100
    for epoch in range(EPOCHS):
        model.train()
        output = model(train_data)
        loss = criterion(output, train_target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            correct=0
            total=0
            output = model(test_data)
            _1,predicted1 = torch.max(output,1) 
            correct += (predicted1==test_target).sum().item()
            total += test_target.size(0)
            accuracy = 100*correct/total
            print(f'epoch:{epoch},loss:{loss},accuracy={accuracy:.2f}%')
            writer.add_scalar("loss", loss, epoch)
            writer.add_scalar("accuracy", accuracy, epoch)
    writer.close()

if __name__ == '__main__':
    #加载数据集
    olivetti_faces = fetch_olivetti_faces(data_home='./olivetti', shuffle=True)
    origin_input = torch.tensor(olivetti_faces.images)
    train_data = origin_input[:300] 
    test_data = origin_input[300:400]
    target = torch.tensor(olivetti_faces.target).long()
    train_target = target[:300]
    test_target = target[300:400]

    #实例化模型
    rnn_model = RNN_Classifier()
    gru_model = GRU_Classifier()
    lstm_model = LSTM_Classifier()
    #定义损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=1e-3)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)

    train(rnn_model,criterion,rnn_optimizer, train_data, train_target, test_data, test_target)
    train(gru_model,criterion,gru_optimizer, train_data, train_target, test_data, test_target)
    train(lstm_model,criterion,lstm_optimizer, train_data, train_target, test_data, test_target)

    
