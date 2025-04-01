# %%
import torch
import numpy as np
arr = np.array([1,2,3], float)
arr

# %% [markdown]
# a=[5,3,7]
# a

# %%
data = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
data

# %%
a=np.zeros((2,4),dtype=float)
print(a)
b=np.ones((3,2),float)
print(b)

# %%
a=np.arange(2,5,1)
print(a)
print()
b=np.arange(1,9,2)
print(b)
print()
c=np.eye(5)
print(c)

# %%
# 指定长度0-1之间的平均分布随机数组
a=np.random.random(5)
print(a)
# 正态分布mu是均值sigma是标准差
mu,sigma=0,0.1
b=np.random.normal(mu,sigma,2)
print(b)
np.random.normal(0,1,4)

# %%
# 数组的切片操作，数组的访问
# 用array()创建数组时，参数必须是方括号括起来的列表
a=np.array([(2,3,1),(2,3,4),(6,5,4)])
print(a)
b=np.array([(),(),()])
print(b)
# 提取前n-1列
print(a[:,:2])
# 从第1列到最后1列
print(a[:,1:])
# 提取前2行
print(a[:2,:])
# 提取后两行
print(a[1:,:])

# %%
# 数组遍历
a=np.array([5,3,2])
for i in a:
    print(i)
# 多维数组的遍历
b=np.array([(2,3,1),(4,2,7),(5,3,6)])
for i,j,k in b:
    print(i+j*k)

# %%
# NumPy数组的常用属性
a = np.array([(2,3,1),(4,2,7),(5,3,6)])
print("ndim维度，秩:",a.ndim)
print("shape大小",a.shape)
print("size元素的总个数",a.size)
print("dtype元素类型",a.dtype)

# %%
# 检测元素是否在数组中
a = np.array([(2,3,1),(4,2,7),(5,3,6)])
print(2 in a)
print(5 in a)
print(9 in a)

# %%
# 数组的重排列
a = np.array([(2,3,1),(4,2,7),(5,3,6)])
a.reshape(9)
print(a.reshape(9))
# 转置
print(a.T)
# 把多维数组转变为一维数组，要求每个元组的长度必须是相同的
print(a.flatten())
print(a.shape)
print()
a=a[:,np.newaxis]#在第2个维度上增加一个新的维度
print(a)
print(a.shape)

# %%
# 矩阵的+-*/
# *是对应位置相乘
a=np.ones((2,2))
b=np.array([(1,1),(2,2)])
print(a)
print(b)
print(a+b)
print(a-b)
print(a*b)
print(a/b)

# %%
a=np.array([2,3,7])
print(a.sum())
print(a.prod())

# %%
# 平均数，方差，标准差，最大值，最小值
a=np.array([4,3,9])
print("mean:",a.mean())
print("var:",a.var())
print("std:",a.std())
print("max:",a.max())
print("min:",a.min())

# %%
# 最大值和最小值对应的索引值,元素值的上限，下限，四舍五入
a=np.array([2.1,1.9,-5.1])
print("a:",a)
print("argmax:",a.argmax())
print("argmin:",a.argmin())
print("ceil:",np.ceil(a))
print("floor:",np.floor(a))
print("rint:",np.rint(a))
# 排序
a.sort()
# a
print(a)

# %%
# a = np.array([16,31,12,28,22,31,48])
# a.sort()
# a
a=np.array([2.1,1.9,-5.1])
a.sort()
a

# %%
# NumPy线性代数
# 矩阵和数组，推荐使用数组，因为数组更灵活
# 矩阵乘法dot计算一维数组是内积，对应位置相乘
import numpy as np
n1=np.array([[1,2],[3,4]],dtype=np.float32)
n2=np.array([[5,6],[7,8]],dtype=np.float32)
result_dot=np.dot(n1,n2)
print(result_dot)
result_at = n1@n2
print(result_at)
# @和np.dot()的结果一样

# %%
# NumPy广播机制
# 如果形状相同，则是数组对应位置相乘
a = np.array([1,2,3])
b = np.array([4,5,6])
print(a+b)
# 如果形状不同，则自动触发广播机制
a=np.array([(1,1),(2,3),(2,2)])
b=np.array([-1,1])
print(a.shape)
print(b.shape)
print(a+b)

# %%
# pytorch复习
# 由Facebook的人工智能研究团队开发的
# 张量是pytorch中的基本单位，把张量看成是一个容器
# 张量可以通过多种方式初始化
# 从数据中创建，数据类型自动推断
import torch
import numpy as np
data = [[1,2],[3,5]]
x_data = torch.tensor(data)
print(x_data)

# 从numpy数组创建
np_array = np.array(data)
x_np=torch.from_numpy(np_array)


print("把tensor转变为numpy:")
print(x_data.numpy())

# 从另外的张量创建
# 新张量保留参数张量的属性,包括形状、数据类型
x_ones = torch.ones_like(x_data)
print(f"ones tensor:\n{x_ones}\n")
x_rand=torch.rand_like(x_data,dtype=torch.float)# 覆盖x_data的数据类型
print(f"random tensor:\n{x_rand}\n")


# %%
# 随机值或者常量值
shape=(2,3,)
rand_tensor = torch.rand(shape)
print(rand_tensor)
ones_tensor=torch.ones(shape)
print(ones_tensor)
zeros_tensor=torch.zeros(shape)
print(zeros_tensor)

# %%
# 用现有的tensor构建，使用新的值填充
m=torch.ones(5,3,dtype=torch.double)
print(m)
n=torch.rand_like(m,dtype=torch.float)
print(n)
print(m.size())
print(n.size())
# 均匀分布
torch.rand(5,3)
# 标准正态分布
torch.randn(5,3)
# 离散正态分布
torch.normal(mean=0,std=1.0,size=(5,3))
# 线性间隔向量（返回一个1维张量，包含在区间start和end上均匀间隔的steps个点）
torch.linspace(start=1,end=10,steps=22)

# %%
# 张量的属性，描述了张量的形状、数据类型、存储它们的设备
# 张量可以看做是具有特征和方法的对象
tensor_n = torch.rand(2,6)
print(f"shape of tensor:{tensor_n.shape}")
print(f"datatype of tensor:{tensor_n.dtype}")
print(f"device tensor is stored on:{tensor_n.device}")


# %%
# 张量运算
# 张量的运算都能在GPU上运行，速度通常比在CPU上更快，默认是在CPU上创建
# 跨设备复制内容量较大的张量，在时间和内存方面的成本会很高
# 把张量设置在GPU上运行
if torch.cuda.is_available():
    tensor_n=tensor_n.to('cuda')


# %%
# 张量的索引和切片
tensor = torch.ones(4,4)
print("tensor:",tensor)
print('first row:',tensor[0])
print('first column:',tensor[:,0])
print('last column1:',tensor[...,-1])
print('last column2:',tensor[:,-1])
tensor[:,1]=0
print(tensor)

# %%
# 张量的拼接
t1=torch.cat([tensor,tensor,tensor],dim=1)
print(t1)
t2=torch.stack([tensor,tensor,tensor],dim=1)
print(t2)

# %%
# 计算两个张量之间矩阵乘法的几种方式
y1=tensor @ tensor.T
y2=tensor.matmul(tensor.T)
#torch.rand_like() 用于生成一个与输入张量形状相同、且元素值服从均匀分布（范围在 [0, 1) 之间）的新张量。
y3=torch.rand_like(tensor)
# 把乘法计算结果保存到y3，避免额外的内存分配
torch.matmul(tensor,tensor.T,out=y3)
print(y1)
print(y2)
print(y3)


# %%
# 计算张量逐元素相乘
z1=tensor*tensor
z2=tensor.mul(tensor)
z3=torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)

# %%
agg=tensor.sum()
print(tensor)
print(agg)
# 把张量聚合计算之后的值转换为python数值
agg_item=agg.item()
print(agg_item,type(agg_item))

# %%
# In-place操作虽然节省了⼀部分内存，但在计算导数时可能会出现
# 问题，因为它会⽴即丢失历史记录。因此，不⿎励使⽤它们。
# 把计算结果存储到当前操作数中的操作就称为就地操作
print(tensor,"\n")
tensor.add_(1)
print(tensor)

# %%
# 张量和numpy之间的转换
# CPU 和 NumPy 数组上的张量共享底层内存位置，所以改变⼀个另⼀个也会变
# 张量到numpy数组
t=torch.ones(4)
print(f"t:{t}")
n=t.numpy()
print(f"n:{n}")
t.add_(1)
print(f"t:{t}")


# %%
# numpy数组到张量
n=np.ones(3)
t=torch.from_numpy(n)
print(t)
# 数组的变化也会反映在张量中
# 数组变张量也会变
np.add(n,1,out=n)
print(f"n:{n}")
print(f"t:{t}")


# %% [markdown]
# 所有的深度学习框架都依赖于计算图来完成梯度下降，优化梯度值等计算
# 计算图的创建和应用，会包含两个部分：
# 1 用户构建前向传播图
# 2 框架处理后向传播(梯度更新)
# pytorch和TensorFlow都使用计算图来完成工作，TensorFlow1.x用的是静态计算图，TensorFlow2.x和pytorch使用的是动态计算图
# 动态图对调试友好，允许逐行执行代码，并可以访问所有张量。前提是要熟练掌握程序的调试技能

# %%
# pytorch计算图可视化
# 用torchviz实现
import torch
from torchviz import make_dot
# 矩阵A，向量b，常数c
A = torch.randn(10,10,requires_grad=True) # randn是标准正态分布
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10,requires_grad=True)

# 计算 x^T * A + b * x + c
result = torch.matmul(x.T,A)+torch.matmul(b,x)+c
# import os
# os.environ["PATH"] += os.pathsep + 'E:/soft/Graphviz/bin'
# 生成计算图节点
dot = make_dot(result,params={'A':A,'b':b,'c':c,'x':x})
# 绘制计算图
dot.render('expression',format='png',cleanup=True,view=False)





