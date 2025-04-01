# 导入PyTorch库
import torch
import numpy as np

# 初始化张量
data = torch.tensor([[2, 3],[4,5]], dtype=torch.float32)
print(data)
print(data.dtype)

# 从已知张量维度创建新张量
data = torch.tensor([[1,2],[3,4]])
# 维度指定同data
# 全一
data1 = torch.ones_like(data)
print(data1)
# 全0
data3 = torch.zeros_like(data)
print(data3)
# 0~1 随机数-小数
data2 = torch.rand_like(data, dtype=torch.float)
print(data2)
# 均值为 0，标准差为 1 - 正态分布
data4 = torch.randn_like(data, dtype=torch.float)
print(data4)
# 1~10 随机数-整数
data5 = torch.randint_like(data, 1, 10)
print(data5)
# 初始化全3的张量
data6 = torch.full_like(data, 3)
print(data6)
# 内容不可预测的随机张量
data7 = torch.empty_like(data)
print(data7)

print("========================================")

# 维度自定的张量
shape = (2,3)
# 均匀分布
data1 = torch.rand(shape)
print(data1)
# 正态分布
data2 = torch.randn(shape)
print(data2)
# 离散正态分布
data3 = torch.normal(mean=.0, std=1.0, size=shape)
print(data3)
# 线性间隔向量-在区间start和end上均匀间隔的steps个点
data4 = torch.linspace(1, 10, steps=5)
print(data4)
# 全一
data4 = torch.ones(shape)
print(data4)

print("========================================")

# 张量属性
tensor = torch.tensor([[1,2],[3,4]])
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print("========================================")

# window-识别张量是否存储在GPU上
tensor = torch.rand(4,4)
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)
else:
    print("No GPU available")
    device = tensor.device
    print("use ", device)

# mac-识别张量是否存储在GPU上
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)
else:
    print("No mps available")
    device = tensor.device
    print("use ",device)
    
print("=======================================")

# 索引和切片
tensor = torch.tensor([[1,2],[3,4]])
print("第二行元素：", tensor[1])
print("第一列元素：", tensor[:,0])
print("第二列元素：", tensor[:,1])
print("最后一列元素：", tensor[...,-1])
print("对角线元素：", tensor.diag())

print("=======================================")

# 张量的拼接
data1 = torch.tensor([[1,2,3],[4,5,6]])
data2 = torch.tensor([[7,8,9],[10,11,12]])
# cat：续接，dim取值[-2, 1], -2=0, -1=1
# dim=0 表示按行拼接-A/B摞在一起
data3 = torch.cat([data1, data2], dim=0)
print(data3)
# dim=1 表示按列拼接-A/B横着放
data4 = torch.cat([data1, data2], dim=1)
print(data4)

# stack：叠加 
# 学习参考：https://blog.csdn.net/flyingluohaipeng/article/details/125034358
# dim取值 0 到输入张量的维数
# 0：左右拼接加维度
data5 = torch.stack((data1, data2), dim=0)
print(data5)
print(data5.shape)
# 1：每个行进行组合
data6 = torch.stack((data1, data2), dim=1)
print(data6)
print(data6.shape)
# 2：对相应行中每个列元素进行组合
data7 = torch.stack((data1, data2), dim=2)
print(data7)
print(data7.shape)

print("=======================================")


# 算数运算
tensor = torch.arange(1, 10, dtype=torch.float32).reshape(3,3)
print(tensor)

# 加法
data1 = tensor + 1
print(data1)   
# 减法 
data2 = tensor - 1
print(data2)
# 乘法
data3 = tensor * 2
print(data3)
# 逐元素相乘
data3 = tensor * tensor.T
print(data3)
y3 = torch.rand_like(tensor)
torch.mul(tensor, tensor.T, out=y3)
print("逐元素相乘", y3)
# 矩阵乘法-内积
data3 = tensor @ tensor.T
print(data3)
data3 = tensor.matmul(tensor.T)
print(data3)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)
# 除法
data4 = tensor / 2
print(data4)

print("=======================================")

# 单元素张量转原生基本类型
agg = tensor.sum()
print(agg, type(agg))
# 原生基本类型
agg_item = agg.item()
print(agg_item, type(agg_item))

print("=======================================")

# 张量转numpy
# np_arr = tensor.numpy()
# print(np_arr, type(np_arr))

print("=======================================")

# 危险函数（计算图中慎用）
# In-place操作(就地操作)-值保存在原变量中
print(tensor, "\n")
print(tensor.add_(5), "\n")
print(tensor)

a_torch = torch.tensor([1,2,3])
# 两者区别：
# 内存占用：a_np = np.array(a_torch)会创建一个新的NumPy数组，占用额外的内存空间。
# 所以a_torch修改，b_np不变
b_np = np.array(a_torch)
# numpy  a_torch修改，c_np变
c_np = a_torch.numpy()

print("=======================================")

import torch
from torchviz import make_dot

# ax + b
a = torch.randn(10, requires_grad=True)
b = torch.randn(10, requires_grad=True)
x = torch.randn(10, requires_grad=True)

y = a * x + b

dot = make_dot(y, params={'a': a, 'b': b, 'x': x})
dot.render('onex', format='png', cleanup=True, view=False)

print("=======================================")

# 定义矩阵 A，向量 b 和常数 c
A = torch.randn(10, 10,requires_grad=True)  # requires_grad=True 表示我们要对 A 求导
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)


# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c

# 生成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)



