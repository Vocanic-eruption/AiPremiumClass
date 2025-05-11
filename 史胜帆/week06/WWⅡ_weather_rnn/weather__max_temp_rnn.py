# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T14:48:08.930873Z","iopub.execute_input":"2025-04-09T14:48:08.931183Z","iopub.status.idle":"2025-04-09T14:48:08.951808Z","shell.execute_reply.started":"2025-04-09T14:48:08.931154Z","shell.execute_reply":"2025-04-09T14:48:08.950853Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:03:22.362972Z","iopub.execute_input":"2025-04-09T15:03:22.363366Z","iopub.status.idle":"2025-04-09T15:03:23.164912Z","shell.execute_reply.started":"2025-04-09T15:03:22.363331Z","shell.execute_reply":"2025-04-09T15:03:23.163815Z"}}
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset,DataLoader

max_temp = []

fixed = open('WWⅡ_max_temp.txt','w',encoding = 'utf-8')
with open('/kaggle/input/weatherww2/Summary of Weather.csv','r',encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for item in reader:
        max_temp_item = item['MaxTemp']
        max_temp.append(round(float(max_temp_item),2))
        

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:08:44.033369Z","iopub.execute_input":"2025-04-09T15:08:44.033818Z","iopub.status.idle":"2025-04-09T15:08:44.040680Z","shell.execute_reply.started":"2025-04-09T15:08:44.033784Z","shell.execute_reply":"2025-04-09T15:08:44.039509Z"}}
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset,DataLoader


#print(max_temp[:10])

def generate_series(max_temp,n_steps,pred_steps):
    
    max_temp = np.array(max_temp,dtype = np.float32)
    
    max_temp = max_temp.reshape(-1,30,1) #如果reshape()不满足整数怎么办
    print(max_temp.shape)  #(1984,60,1)
    
    
    X_train,y_train = max_temp[:1500,:n_steps,:],max_temp[:1500,pred_steps:,:]
    x_test,y_test = max_temp[1500:1984,:n_steps,:],max_temp[1500:1984,pred_steps:,:]
    
    print(X_train.shape,y_train.shape)
    print(x_test.shape,y_test.shape)

    return X_train,y_train,x_test,y_test

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:12:48.635344Z","iopub.execute_input":"2025-04-09T15:12:48.635753Z","iopub.status.idle":"2025-04-09T15:12:48.650350Z","shell.execute_reply.started":"2025-04-09T15:12:48.635720Z","shell.execute_reply":"2025-04-09T15:12:48.649107Z"}}
X_train,y_train,x_test,y_test = generate_series(max_temp,29,-1)

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:12:50.730836Z","iopub.execute_input":"2025-04-09T15:12:50.731161Z","iopub.status.idle":"2025-04-09T15:12:50.736182Z","shell.execute_reply.started":"2025-04-09T15:12:50.731131Z","shell.execute_reply":"2025-04-09T15:12:50.734811Z"}}
# 模型参数
EPOCHES = 40
LR = 0.009
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 4


# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:12:52.425932Z","iopub.execute_input":"2025-04-09T15:12:52.426327Z","iopub.status.idle":"2025-04-09T15:12:52.434309Z","shell.execute_reply.started":"2025-04-09T15:12:52.426292Z","shell.execute_reply":"2025-04-09T15:12:52.433264Z"}}
class WeatherSeriesDataset(Dataset):
    def __init__(self,X,y = None,train = True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self,ix):
        if self.train:
            return torch.from_numpy(self.X[ix]),torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])

dataset = {
    'train':WeatherSeriesDataset(X_train,y_train),
    'test':WeatherSeriesDataset(x_test,y_test)

}

dataloader = {
    'train':DataLoader(dataset['train'],batch_size = BATCH_SIZE_TRAIN,shuffle = True),
    'test':DataLoader(dataset['test'],batch_size = BATCH_SIZE_TEST,shuffle = False)

}

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:12:54.996157Z","iopub.execute_input":"2025-04-09T15:12:54.996524Z","iopub.status.idle":"2025-04-09T15:12:55.004590Z","shell.execute_reply.started":"2025-04-09T15:12:54.996495Z","shell.execute_reply":"2025-04-09T15:12:55.003178Z"}}
# 预测单个天气 rnn
class RNN_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size = 1,
                          hidden_size = 10,
                          bias = True,
                          batch_first = True,
                          num_layers = 1
                          #dropout = 0.20
        )
        self.fc = nn.Linear(in_features = 10,out_features = 1,bias = True)

    def forward(self,X):
        out,_ = self.rnn(X)
        out = self.fc(out[:,-1,:])
        return out

model = RNN_Classifier()


        

# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:12:58.272851Z","iopub.execute_input":"2025-04-09T15:12:58.273182Z","iopub.status.idle":"2025-04-09T15:12:58.278125Z","shell.execute_reply.started":"2025-04-09T15:12:58.273156Z","shell.execute_reply":"2025-04-09T15:12:58.276901Z"}}
# 损失函数
cerition = nn.MSELoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(),lr = LR)


# %% [code] {"execution":{"iopub.status.busy":"2025-04-09T15:13:00.075971Z","iopub.execute_input":"2025-04-09T15:13:00.076484Z","iopub.status.idle":"2025-04-09T15:13:29.297978Z","shell.execute_reply.started":"2025-04-09T15:13:00.076438Z","shell.execute_reply":"2025-04-09T15:13:29.296938Z"}}
# 模型训练

model.train()
for epoch in range(EPOCHES):
    for data,target in dataloader['train']:
        optimizer.zero_grad()
        output = model(data)
        loss = cerition(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1.0)
        optimizer.step()
    print(f'epoch:{epoch + 1},loss:{loss.item()}')
    
