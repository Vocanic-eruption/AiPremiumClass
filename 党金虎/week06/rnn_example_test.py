import torch
rnn = torch.nn.RNN(input_size=10, hidden_size=20, batch_first=True)
inputs = torch.randn(3, 5, 10)  # 3个样本，序列长度5，特征维度10
out, h_n = rnn(inputs)
print("输出形状:", out.shape)  # (3,5,20)
print("最后一个隐藏状态:", h_n.shape)  # (1,3,20)

# 训练和测试集
data = [1,2,3,4,5,6,7,8,9]
X,y = [],[]
step = 2
for i in range(len(data)-step):
    X.append(data[i:i+step,0])

print(X)
    
    
