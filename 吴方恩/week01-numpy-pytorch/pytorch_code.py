import torch
import numpy as np

# 张量创建
data1 = torch.tensor([[1,1], [2,2]],dtype=torch.float32)
print(data1)

data2 = np.array([[1,1], [2,2]])
data3 = torch.from_numpy(data2)
print(data3)

data4 = torch.rand_like(data1)
print(data4)

shape = (3,4)
data5 = torch.rand(shape)
data6 = torch.zeros(shape)
data7 = torch.ones(shape)

print(f"Random Tensor: \n {data5} \n")
print(f"Zeros Tensor: \n {data6}")
print(f"Ones Tensor: \n {data7} \n")


# 基于现有tensor构建，但使用新值填充
data8 = torch.ones(2,3,dtype=torch.double)
data9 = torch.rand_like(data8, dtype=torch.float32)
print('张量大小:', data9.size())

# 均匀分布
print(torch.rand(3,4))
# 标准正态分布
print(torch.randn(3,4))
# 离散正态分布
print(torch.normal(mean=.0,std=1.0,size=(3,4)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=4))

# 张量属性
print('张量大小:\n', data9.shape)  
print('张量元素个数:\n', data9.numel())
print('张量维度:\n', data9.dim())
print('张量元素类型:\n', data9.dtype)
print('张量设备:\n', data9.device)

# mac上没有GPU，使用M系列芯片
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = torch.tensor([1,2,3]).to(device)

print(tensor)
print(tensor.device)

# 张量索引和切片
data10 = torch.rand(4,3)
print(data10)
print('第1行' ,data10[0]) # 第1行
print('第1行第2列',data10[0][1]) # 第1行第2列
print('第1行第2列',data10[0,1]) # 第1行第2列
print('第1行',data10[0,:]) # 第1行
print('第1列',data10[:,0]) # 第1列
print('第1行到第2行',data10[0:2,:]) # 第1行到第2行
print('第1行到第2行',data10[:2,:]) # 第1行到第2行
print('第1行到第2行',data10[:-1,:]) # 第1行到第3行
print('最后一列',data10[:,-1]) # 最后一列

# 张量运算
data11 = torch.cat([torch.tensor([[1,2,3]]), torch.tensor([[3,2,1]]), torch.tensor([[2,1,3]])], dim=1)
print(data11 * 3)
print(data11.shape)

data12 = torch.arange(1,5, dtype=torch.float32).reshape(2, 2)

# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = data12 @ data12.T
y2 = data12.matmul(data12.T)

# print(y1)
# print(y2)

y3 = torch.rand_like(data12)
torch.matmul(data12, data12.T, out=y3)
# print(y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = data12 * data12
z2 = data12.mul(data12)

z3 = torch.rand_like(data12)
torch.mul(data12, data12, out=z3)

print(z1)
print(z3)

# In-place操作
data13 = torch.tensor([[1,2,3], [4,5,6]])
data13.add_(1)
print(data13)


# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
