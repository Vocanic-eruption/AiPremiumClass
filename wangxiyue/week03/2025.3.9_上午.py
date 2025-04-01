import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from sklearn.datasets import load_iris

#数据准备
x,y = load_iris(return_X_y=True)
x=x[:100]
y=y[:100]
# 创建张量
tensor_x  = torch.tensor(x,dtype=torch.float32)
tensor_y = torch.tensor(y,dtype=torch.float32)

#参数设置
learning_rate = 0.01
epochs = 500
#模型参数
w = torch.randn(1,4 , requires_grad=True) #gradient=true 表示该张量需要 梯度计算， w.grad 默认值none 保存梯度值
b = torch.randn(1 , requires_grad=True) # bias

for i in range(epochs):
    #前向计算
    z = torch.nn.functional.linear(tensor_x,w,b)
    z = torch.sigmoid(z)
    #损失函数
    # z.squeeze() 去除张量为 1 的维度 等价于 z.reshape(-1)
    loss = torch.nn.functional.binary_cross_entropy(z.squeeze(),tensor_y,reduction='mean') # reduction = 'mean' 求平均
    #计算梯度 , 求导过程，在 pytorch中计算求导过程
    loss.backward() # 计算梯度、梯度值保存在 ，w.grad和b.grad
    # print(f"w.grad = {w.grad} , b.grad = {b.grad}")

    # 参数更新
    with torch.no_grad(): # 关闭梯度计算跟踪
        # 这里不能用 w = w - learning_rate*w.grad
        # 会把 requires_grad=True属性丢掉
        # 可以通过 print(f'w = {w}')，print(f'b = {b}')观察
        w -= learning_rate*w.grad
        b -= learning_rate*b.grad
        w.grad.zero_() # clean 梯度
        b.grad.zero_() # clean 梯度
    #训练动态损失
    if i % 100 == 0:
        print(f'train loss {loss.item()}')






# for x,y in test_loader :
#     print("x.shape [N,C,H,W] = ",x.shape)
#     print("y.shape =" , y.shape,y.dtype)

