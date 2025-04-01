import numpy as np

# 创建ndarray数组
arr = np.array([5,6,7,8,9])
print(arr)
print(type(arr))

arr2 = np.array([[1,2], [4,5]])
print(arr2)

# 创建全0数组
arr3 = np.zeros((3,4))
print('创建全0数组', arr3)

# 创建全1数组
arr4 = np.ones((3,4))
print('创建全1数组', arr4)

# 根据范围创建数组
arr5 = np.arange(10,20,2)
print('根据范围创建数组', arr5)

# 创建对称矩阵
arr6 = np.eye(3)
print('创建对称矩阵', arr6)

# 创建随机数组
arr7 = np.random.random((2,3))
print('创建随机数组:\n', arr7)

# 创建正态分布随机数
arr8 = np.random.normal(3,0.1,4)
print('创建正态分布随机数:\n', arr8)

# 数组切片操作
arr9 = np.array(([1,2,3], [4,5,6], [7,8,9], [10,11,12]))
print('数组切片操作 [1:2, 1:3]:\n', arr9[1:2, 1:3]) #1:2, 1:3 表示行1到2，列1到3
print('数组切片操作 [:4, 1:3]:\n', arr9[:4, 1:3]) #:4, 1:3 b表示行0到3，列1到2

# 数组的属性
arr10 = np.array([[1,2,3], [4,5,6]])
print('数组的属性[形状]:\n', arr10.shape)
print('数组的属性[元素个数]:\n', arr10.size)
print('数组的属性[维度]:\n', arr10.ndim)
print('数组的属性[元素类型]:\n', arr10.dtype)

# 数组形状改变
arr11 = np.array([[1,2,3], [4,5,6]])
print('数组形状改变前:\n', arr11)
arr12 = arr11.reshape((3,2))
print('数组形状改变后:\n', arr12)

# 数组扁平化
arr13 = np.array([[1,2,3], [4,5,6]])
print('数组扁平化前:\n', arr13)
arr14 = arr13.flatten()
print('数组扁平化后:\n', arr14)
# 数组转置
arr15 = np.array([[1,2,3], [4,5,6]])
print('数组转置前:\n', arr15)   
arr16 = arr15.T
print('数组转置后:\n', arr16)
# 数组增加维度
arr17 = np.array([1,2,3,4,5,6])
print('数组增加维度前:\n', arr17.shape)
arr18 = arr17[:,np.newaxis]
print('数组增加维度后:\n', arr18.shape)

# 数组运算
arr19 = np.array([[1,2,3], [4,5,6]])
arr20 = np.ones((2,3))
print('数组相加:\n', arr19 + arr20) #[2,3,4],[5,6,7]
print('数组相减:\n', arr19 - arr20) #[0,1,2],[3,4,5]
print('数组相乘:\n', arr19 * arr20) #[1,2,3],[4,5,6]
print('数组相除:\n', arr19 / arr20) #[1,2,3],[4,5,6]

# 数组函数
arr21 = np.array([[1,2,3], [4,5,6]])
print('求和:\n', np.sum(arr21)) #21
print('求积和:\n', np.prod(arr21)) #720
print('求平均值:\n', np.mean(arr21)) #3.5
print('求最大值:\n', np.max(arr21)) #6
print('求最小值:\n', np.min(arr21)) #1
print('求标准差:\n', np.std(arr21)) #1.707825127659933  
print('求方差:\n', np.var(arr21)) #2.9166666666666665
print('求最大值索引:\n', np.argmax(arr21)) #5
print('求最小值索引:\n', np.argmin(arr21)) #0

# numpy线性代数
arr22 = np.array([[1,2], [3,4]])    
arr23 = np.array([[5,6], [7,8]])
print('矩阵相乘:\n', np.dot(arr22, arr23)) #[[19,22],[43,50]]
print('矩阵相乘:\n', arr22 @ arr23) #[[19,22],[43,50]]
# [[1,2],
#  [3,4]]
# [[5,6],
#  [7,8]]
# [[1*5+2*7,1*6+2*8],
#  [3*5+4*7,3*6+4*8]]
# [[19,22],[43,50]]
print('矩阵转置:\n', np.transpose(arr22)) #[[1,3],[2,4]]

# numpy广播机制
arr24 = np.array([[1,2], [3,4]])
arr25 = np.array([1,2])
print('arr24的形状:\n', arr24.shape) #(2,2)
print('arr25的形状:\n', arr25.shape) #(2,)
# 广播机制：将arr25的形状(2,)变成(1,2)，再变成（2,2）然后进行相加
print('广播机制:\n', arr24 + arr25) #[[2,4],[4,6]]

# numpy 保存和加载
arr26 = np.array([[1,2], [3,4]])
np.save('../arr26.npy', arr26)
arr27 = np.load('../arr26.npy')
print('加载数组:\n', arr27)

