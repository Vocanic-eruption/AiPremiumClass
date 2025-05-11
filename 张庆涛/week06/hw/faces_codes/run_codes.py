import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from Fetch_Olivetti_Faces_model import FetchOlivettiFaces,OlivettiDataset
from torch.utils.tensorboard import SummaryWriter


# 加载数据集
def data_load():
    # 使用 fetch_olivetti_faces 函数加载数据，设置数据存储路径为 '../data'，并打乱数据，返回特征和标签
  olivetti_faces= fetch_olivetti_faces(data_home='../data',shuffle=True)
    # 将标签转换为 PyTorch 张量
  targets = torch.tensor(olivetti_faces.target)
  # 创建 OlivettiDataset 实例，传入数据和标签
  dataset = OlivettiDataset(olivetti_faces.data, targets)
    # 返回创建的 dataset 对象
  return dataset

# 拆分数据集 
def split_dataset(dataset):
  # 计算数据集大小
  dataset_size = len(dataset)
  # 计算训练集大小
  train_size = int(dataset_size * 0.6) # 60% 
  # 计算测试集大小 
  test_size = int(dataset_size * 0.2) # 20%
  # 计算验证集大小
  valid_size = dataset_size - train_size-test_size # 20%
  # 划分数据集
  train_data, test_data,valid_data,  = torch.utils.data.random_split(dataset, [train_size, test_size,valid_size, ])
  return train_data, test_data,valid_data

# 另定义数据集
def train_model(epochs,model,train_loader,valid_loader,criterion,optimizer,writer,model_name):
  best_valid_accuracy = 0  # 记录最佳验证准确率
  for epoch in range(1,epochs+1):
    model.train()
    total_loss = 0  # 记录当前 epoch 的总损失
    correct = 0  # 记录当前 epoch 的正确预测数量
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
      optimizer.zero_grad() # 清空梯度
      outputs  = model(images) # 前向传播,得到模型的输出
      loss = criterion(outputs , labels) # 计算损失
      loss.backward() # 反向传播 计算梯度
      optimizer.step() # 更新参数
      total_loss += loss.item() # 累加损失
      _, predicted = torch.max(outputs, 1) # 获取预测结果
      correct += (predicted == labels).sum().item() # 累加正确预测数量
    train_accuracy = correct / len(train_loader.dataset)  # 计算训练集准确率
    writer.add_scalar(f'Loss/train/{model_name}', total_loss / len(train_loader), epoch)  # 记录训练损失
    writer.add_scalar(f'Accuracy/train/{model_name}', train_accuracy, epoch)  # 记录训练准确率
    print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss / len(train_loader):.5f}, Accuracy: {train_accuracy:.5f}') # 打印训练损失和准确率
    
    # 验证模型
    model.eval()
    with torch.no_grad():
      correct = 0
      for images, labels in tqdm(valid_loader, desc="Validating"):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
      valid_accuracy = correct / len(valid_loader.dataset)  # 计算验证集准确率
      writer.add_scalar(f'Accuracy/valid/{model_name}', valid_accuracy, epoch)  # 记录验证准确率
      print(f'Validation Accuracy: {valid_accuracy:.5f}')
      # 更新最佳验证准确率
      if valid_accuracy > best_valid_accuracy:
          best_valid_accuracy = valid_accuracy
  return best_valid_accuracy
# 测试模型
def test_model(model,test_loader):
  model.eval()
  with torch.no_grad():
    correct = 0
    for images, labels in tqdm(test_loader, desc="Testing"):
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      correct += (predicted == labels).sum().item()
    test_accuracy = correct / len(test_loader.dataset)  # 计算测试集准确率
    print(f'Test Accuracy: {test_accuracy:.5f}')
  return test_accuracy


if  __name__ == '__main__':
  
  # 设置超参数
  BATCH_SIZE = 128
  HIDDEN_SIZE = 256
  EPOCHS  = 40
  LR = 0.001


  # 加载数据集
  dataset = data_load()
  # 拆分数据集
  train_data, test_data,valid_data = split_dataset(dataset)
  # 创建数据加载器
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
  valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
  # 定义模型列表
  models = {
    'RNN': FetchOlivettiFaces(input_size=4096, hidden_size=HIDDEN_SIZE, num_classes=40,num_layers=3, module='RNN'),
    'GRU': FetchOlivettiFaces(input_size=4096, hidden_size=HIDDEN_SIZE, num_classes=40,num_layers=3, module='GRU'),
    'LSTM': FetchOlivettiFaces(input_size=4096, hidden_size=HIDDEN_SIZE, num_classes=40,num_layers=3, module='LSTM'),
    'BiRNN': FetchOlivettiFaces(input_size=4096, hidden_size=HIDDEN_SIZE, num_classes=40,num_layers=3, module='BiRNN'),
    }
  
  # 训练和验证不同的模型
  writer = SummaryWriter()  # 初始化 TensorBoard 记录器
  model_results ={} # 存储每个模型的性能结果
  for model_name, model in models.items():
    print(f"Training {model_name}...")
    criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 定义优化器
    # 检查 epochs
    best_valid_accuracy = train_model(EPOCHS,model, train_loader, valid_loader, criterion, optimizer, writer, model_name)
    test_accuracy = test_model(model, test_loader)  # 测试模型
    model_results[model_name] = {
        'best_valid_accuracy': best_valid_accuracy,
        'test_accuracy': test_accuracy,
        }

   # 打印模型性能总结
  print("\nModel Performance Summary:")
  for model_name, results in model_results.items():
    print(f"{model_name}:")
    print(f"  Best Validation Accuracy: {results['best_valid_accuracy']:.4f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.4f}")

  writer.close()  # 确保日志文件正确关闭