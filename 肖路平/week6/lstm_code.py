import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class LSTM_Classfier(nn.Module):

    def __init__(self,):
        super().__init__()        
        self.lstm = nn.LSTM(
            input_size=64,      #x的特征维度
            hidden_size=50,     #隐层神经元数量
            bias=True,          #偏置[50]
            num_layers=5,      #隐藏层数量            
            batch_first=True    #批次是输入的第一个维度
        )
        self.fc = nn.Linear(50,40) #输出层，50个神经元，40个分类
    
    def forward(self,x):
        x = x.view(-1, 64, 64) 
        # 输入X的shape为[batch,times,features]
        outputs,l_h = self.lstm(x)
        #取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:]) #[batch,hidden_size] 
        return out
    

if __name__ == '__main__':
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    olivetti_faces = fetch_olivetti_faces(data_home='./face_data',shuffle=True)

    data  = torch.tensor(olivetti_faces.data,dtype=torch.float32)
    target  = torch.tensor(olivetti_faces.target,dtype=torch.long)
    #train_dataset = MNIST(root='./data',train=True,transform=ToTensor(),download=True)
    #test_dataset = MNIST(root='./data',train=False,transform=ToTensor(),download=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)

    model = LSTM_Classfier()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)



    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i,(images,labels)  in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images) #去掉一维的维度
            loss = criterion(outputs,labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) #梯度裁剪
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'epoch:{epoch+1}/{num_epochs},loss:{loss.item():.4f}')
                writer.add_scalar('training loss222',loss.item(),epoch*len(train_loader)+i)
        
        
        #评估模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images,labels in test_loader:
                images,labels = images.to(device),labels.to(device)
                outputs = model(images.squeeze())
                _,predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
            writer.add_scalar('Test Accuracy', accuracy, epoch)
    writer.close()