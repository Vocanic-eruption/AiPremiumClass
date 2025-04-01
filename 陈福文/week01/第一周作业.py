# %%
import numpy as np
 

# %%
#创建ndarray数组对象
arr = np.array([4,5,6,7,8,9])
np.array([])
arr

# %%
arr = np.array([10,20.39,55],float)
arr

# %%
#numpy特殊数组的创建
#创建初始化的数组，一般是1和0
a1 = np.zeros((4,5),dtype=np.double)
a1
a1 = np.zeros((4,5),dtype=np.float32)
a1

# %%
a1 = np.ones((2,3),dtype=np.float64)
a1

# %%
#numpy 创建序列
#等差数列,1-5递增，步长=5，左闭右开
a = np.arange(1,5,0.5)
a

# %%
#创建对角为1
np.eye(3)

# %%
#随机数,模型运算参数初始值
np.random.random(5)
#正态分布，normal正态的意思
np.random.normal(0,0.1,5)


# %%
# 数组的创建
np.array([1,2,4])
np.ones((2,3),dtype=np.float32)
np.zeros((1,2),dtype=np.float64)
np.arange(1,10,0.4)
np.eye(4)
np.random.random(5)
np.random.normal(0,1.2,5)


# %%
#Numpy数组的访问
#Numpy可以对高维数组进行操作
a = np.array([(1,2,7),(3,4,8),(5,6,9)])
b = np.array([(1,2),(3,4),(5,6)])
# print(a[0,2])
# print(a[1:])
print(b[:1])
print(b[1:])
print(b[:,:1])
print(b[0,:])
print(a[:,2])
print(a[:,1])


# %%
#numpy数组的遍历
for i in a:
    print(i)

# %%
for i,j,k in a:
    print(i)
    print(j)
    print(k)
    print("====")

# %%
# numpy数组的常用属性(没有括号)
a = np.array([(1,2,3),(4,5,6),(7,8,9)])
#数组的纬度
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)

# %%
# 检测元素是否存在
print(8 in a)

# %%
print(11 in a)

# %%

print((1,2,3) in a)

# %%
# 确保数据总理不变的情况下，改变其形状（纬度）
a = np.arange(1,10)
print(a)
a.reshape   

# %%
b = a.reshape(3,3) #纬度大小的橙剂 =元素个数
b = a.reshape(3,2,1) #是不行的

# %%
#加载图像数据 
#纬度的含义
#img.shape(1,3,120,120) 一个样本，3颜色特征通道，120高，120宽

# %%
# 翻转,转置，神经元运算，回归运算
a = np.array([(1,2,3),(4,5,6),(7,8,9),(11,12,13)])
print(a)
a.transpose()

# %%
a.T

# %%
#将多维转为一维
a.flatten()

# %%
#加减乘除
a = np.ones((2,2))
b = np.array([(-1,1),(-1,1)])
print(a)
print(b)
print(a+b)

# %%
a = np.array([1,2,1])
print(a)
#求和
print(a.sum())
#求乘积
print(a.prod())
print(a.mean())
print(a.var())
print(a.std())
print(a.max())
print(a.min())

# %%
#  索引位置
a = np.array([1,2.2,3.8,5,6,7])
print(a.argmax())
print(a.argmin())
#向上取整
print(np.ceil(a))
#向下取整
print(np.floor(a))
#四舍五入
print(np.rint(a))

# %%
#排序
a.sort()
a

# %%
a[::-1]

# %%
np.sort(a)

# %%
np.sort(a)[::-1]

# %%
#线性代数
#有关线性代数的运算均在numpy.linalg中
#建议用array，更灵活

# %%
#矩阵乘法
#维度结构一样的，按位想乘
a = np.array([[1,2],[3,4]],dtype=np.float32)
b = np.array([[5,6],[7,8]],dtype=np.float32)

res_dot = np.dot(a,b)
res_at = a @ b
print(res_dot)
print(res_at)

# %%
 #创建一个矩阵
manua_res = np.zeros((a.shape[0],b.shape[1]),dtype=np.float32)
print(a.shape[0])
print(manua_res)

# %%
manua_res = np.zeros((2,2),dtype=np.float32)
print(manua_res)

# %%
for i in range(a.shape[0]):
    for j in range(b.shape[1]):
        manua_res[i,j]=0
        for k in range(a.shape[1]):
            print(f"{a[i, k]} * {b[k, j]} = {a[i, k] * b[k, j]}")
            manua_res[i, j] += a[i, k] * b[k, j]
        print(f"结果矩阵[{i+1},{j+1}]:{manua_res[i, j]}\n")
print("手动推演结果:")
print(manua_res)

# %%
#文件操作
np.save('res.npy',manua_res)

# %%
res_load = np.load('res.npy')
print(res_load)

# %%
#numpy的广播机制
#形状不同时，会把低维的扩充和高纬的一样
a = np.array([1,2,3])
b = np.array([4,5,6])
a + b

# %%
a = np.array([[(1,2), (2,2), (3,3), (4,4)]])
print(np.shape(a))
b = np.array([-1,1])
print(np.shape(b))
a+b
b+a

# %%
#升维处理，从低到高，倒一维对应倒二维？


========================pytorch===================
# %%
import torch
import numpy as np

#张量（Tensor）== NdArray
#张量的创建
data = torch.tensor([[1,2],[3,4]])
data

# %%
np_arr = np.array([[1,2],[3,4]])
data = torch.from_numpy(np_arr)
data

# %%
#通过已知张量纬度，创建新张量
data3 = torch.ones_like(data)
data3

# %%
data4 = np.ones_like(data)
data5 = np.ones((1,2))
data5

# %%
data = torch.rand_like(data3,dtype=torch.float32)

# %%
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"1: \n{rand_tensor}\n")
print(f"2: \n{ones_tensor}\n")
print(f"3: \n{zeros_tensor}\n")

# %%
m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
print(m.size()) 
print(torch.rand(5,3))
print(torch.randn(5,3))
print(torch.linspace(start=1,end=20,steps=22))

# %%
tensor = torch.rand(3,4)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

# %%
#是否支持GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# %%
#张量的索引和切片
#和numpy一样
tensor = torch.tensor([(1,2,10),(3,4,12),(5,6,13),(7,8,14)])
print(tensor[1])


# %%
print(tensor[:1])

# %%
print(tensor[:,1])

# %%
#张量的拼接
t1 = torch.cat([tensor,tensor,tensor],dim=0)
print(t1)
print(t1.shape)

# %%
t2 = torch.stack([tensor,tensor,tensor],dim=1)
print(t2)

# %%
tensor = torch.arange(1,10,dtype=torch.float32).reshape(3,3)
print(tensor)

y1 = tensor @ tensor.T

# %%
y2 =tensor.matmul(tensor.T)
print(y1)
print(y2)

# %%
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(z1)
print(z2)
print(z3)

# %%
agg =tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item)

# %%
np_arr = z1.numpy()
np_arr

# %%
#pytorch计算图
#对应的数学模型转换成计算机模型

# %%
A = torch.randn(10,10,requires_grad=True)
print(A)
B = torch.randn(10,requires_grad=True)
print(B)
C = torch.randn(1,requires_grad=True)
print(C)
X = torch.randn(10,requires_grad=True)

# %%
result = torch.matmul(A,X.T)+torch.matmul(B,X)+C
from torchviz import make_dot
dot = make_dot(result,params={'A':B,'B':B,'C':C,'X':X})
dot.render('expression',format='png',cleanup=True,view=False)




