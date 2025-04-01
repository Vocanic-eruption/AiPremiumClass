#成绩单是非得失与
import numpy as np
import torch
from torchviz import make_dot



#类型
a = [1,2,3]
b = np.array(a)
b.dtype

a = [1.2,2,3]
b = np.array(a)
b.dtype

#随机生成特殊数据
a=np.random.random(5)
a=np.random.normal(1,2,5)


#切片
a = np.array([(1,2,3,4), (3,4,5,6), (5,6,7,8),(9,9,9,9)])
a[0]
a[1]

a[0:2]
a[0:3]
a[:,1:2]
a[1:3,1:3]
# for i,j in a[1:3,1:3]:
#         print(i)
#         print('---')
#         print(j)
#         print('***')

a[...,1:]

#属性
a.ndim
a.shape
a.size
a.dtype


#查询
a = np.array([(1,2.0), (3,4)])
#print(2 in a)
#print(3.0 in a)

#展开
a = np.array([(1,2,3,4), (3,4,5,6), (5,6,7,8),(9,9,9,9)])
a.reshape(16)

#转置
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
a.transpose()

a = np.array([(1,2,3)])
a.transpose()
a.T


#维度转化
a = np.array([(1,2), (3,4), (5,6)])
a.flatten()

a = a[ :, np.newaxis]
a = np.array([[1,2], [3,4], [5,6]])
a = np.array([(1,2,3), (4,5,6), (7,8,9)])
a


#加减乘除
b0 = np.array([(-1,1),(0,1)])
#print('b0:',b0.dtype)
b1 = np.array([(-1,1.1),(0,1)])
#print('b1:',b1.dtype)
b2 = np.array([(1,2),(3,4)])
b1+b2

b0 = np.array([(-1,1)])
b1 = np.array([(1,2),(3,4)])
b0+b1

# b0 = np.array([(-1,1),(2,2)])
# b1 = np.array([(1,2),(3,4),(6,8)])
# b0+b1

b0*b1

b1/b0
b0/b1
b=b0.sum()
b=b1.sum()
b

#其他算数
#print("mean:",b1.mean())
#print("var:", b1.var())
#print("std:", b1.std())

#取整
a = np.array([999.99, 2.00001, 4.9,4.5,4.4])
#print("argmax:", a.argmax())
#print("argmin:", a.argmin())
#print("ceil:", np.ceil(a))
#print("floor:", np.floor(a))
#print("rint:", np.rint(a))


#矩阵相乘
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
#print("m1@m2:",m1 @ m2)


m1 = np.array([[1, 2]])
m2 = np.array([[5, 6], [7, 8]])
#print("m1@m2:",m1 @ m2)


m2 = np.array([[5, 6], [7, 8]])
m1=m2.T
#print("m1@m2:",m1 @ m2)


#torch
data = [ 1, 2],[3, 4]
a = torch.tensor(data)
a.dtype

torch.rand(5,3)
torch.randn(5,3)


torch.normal(mean=100,std=9.0,size=(3,3))

torch.linspace(start=100,end=10,steps=20)

#拼接
data = [ 1, 2],[3, 4],[5,6],[7,8],[9,10]
a = torch.tensor(data)
t1 = torch.cat([a, a, a], dim=1)
t1 = torch.cat([a, a, a], dim=0)

data = [ 1, 2],[3, 4]
a = torch.tensor(data)
#print("a*a:",a*a)
#print("a*a:",a@a)

a = a.sum()
print(a.item())


A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
# 计算 x^T * A + b * x + c
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
# ⽣成计算图节点
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)

