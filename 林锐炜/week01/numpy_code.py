import numpy as np

def numpy_base():
    # 数组创建及基础属性
    # 一维数组
    arr_1 = np.array([1, 2, 3], float)
    print(f"一维数组：{arr_1}")
    print(f"数组类型：{type(arr_1)}")
    print(f"维数ndim：{arr_1.ndim}")
    print(f"数组项类型dtype：{arr_1.dtype}")
    print(f"元素个数size:{arr_1.size}")
    print("\n")

    # 二维数组
    arr_2 = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"二维数组：{arr_2}")
    print(f"数组类型：{type(arr_2)}")
    print(f"维数ndim：{arr_2.ndim}")
    print(f"数组项类型dtype：{arr_2.dtype}")
    print(f"元素个数size:{arr_2.size}")
    print("\n")

    # 三维数组
    arr_3 = np.array([[[1, 2, 3], [5, 6, 7]], [[11, 22, 33], [44, 55, 66.]]])
    print(f"三维数组：{arr_3}")
    print(f"数组类型：{type(arr_2)}")
    print(f"维数ndim：{arr_3.ndim}")
    print(f"数组项类型dtype：{arr_3.dtype}")
    print(f"元素个数size:{arr_3.size}")


def inner_func():
    # 内置函数创建数组
    # 创建数组
    print(np.arange(0, 10, 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # zeros
    print("zeros:\n", np.zeros((3, 3), dtype=float))
    """
    [[0. 0. 0.]
    [0. 0. 0.]
    [0. 0. 0.]]
    """

    # ones
    print("ones:\n", np.ones((2, 2), dtype=int))
    """
    [[1 1]
    [1 1]]
    """

    # zeros_like
    print("zeros_like:\n", np.zeros_like([[1, 1], [1, 1]]))
    """
    [[0 0]
    [0 0]]
    """

    # eye
    print("eye:\n", np.eye(4))
    """
    [[1. 0. 0. 0.]
    [0. 1. 0. 0.]
    [0. 0. 1. 0.]
    [0. 0. 0. 1.]]
    """


def const_func():
    # NumPy常用函数
    arr = np.array([1, 2, 3, 4, 5])
    print("求和:", np.sum(arr))  # 15
    print("求均值", np.mean(arr))  # 3.0
    print("最大值:", np.max(arr))  # 5
    print("最小值:", np.min(arr))  # 1
    print("累积和:", np.cumsum(arr))  # [ 1  3  6 10 15]

    arr_2 = np.array([[1, 2], [3, 4], [5, 6]])
    print("二维求和", np.sum(arr_2))  # 21
    print("二维求均值", np.mean(arr_2))  # 3.5
    print("二维最大值:", np.max(arr_2))  # 6
    print("二维最小值:", np.min(arr_2))  # 1
    print("二维累积和:", np.cumsum(arr_2))  # [ 1  3  6 10 15 21]


def index_and_slice():
    # 数据操作
    # 数组索引与切片
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(arr[0])  # [1,2,3]
    print(arr[0:1])  # [[1 2 3]]
    print(arr[0:1, 0])  # [1]
    print(arr[0:1, 0:1])  # [[1]]
    print("\n")

    # bool索引，布尔索引用于基于条件来选择数组中的元素
    arr = np.array([1, 2, 3, 4, 5, 6])
    bool_idx = arr > 3
    print(bool_idx)  # [False False False  True  True  True]
    print(arr[bool_idx])  # 去除大于3的项 [4 5 6]

    # 花式索引，花式索引允许我们使用数组或列表来指定索引顺序，从而按特定顺序选择数组中的元素
    arr = np.array([10, 20, 30, 40, 50])
    indices = [0, 3, 4]
    print(arr[indices])  # [10 40 50]


def numpy_shape():
    # 数组形状 share
    arr_1 = np.array([1, 2, 3], float)
    arr_2 = np.array([[1, 2], [3, 4], [5, 6]])
    arr_3 = np.array([[[1, 2, 3], [5, 6, 7]], [[11, 22, 33], [44, 55, 66.]]])
    print("share...")
    print(f"arr_1 share:{arr_1.shape}")
    print(f"arr_2 share:{arr_2.shape}")
    print(f"arr_3 share:{arr_3.shape}")
    # 转置
    print("转置T...")
    print(f"arr_1 T:{arr_1.T}")
    print(f"arr_2 T:{arr_2.T}")

    print("转置 transpose")
    print(f"transpose 1:{arr_1.transpose()}")
    print(f"transpose 2:{arr_2.transpose()}")

    # reshape 方法可以改变数组的形状而不改变数据内容
    arr = np.array([1, 2, 3, 4, 5, 6])
    reshaped_arr = arr.reshape((3, 2))
    print(reshaped_arr)
    """
    [[1 2]
    [3 4]
    [5 6]]
    """
    reshaped_arr = arr.reshape((2, 3))
    print(reshaped_arr)
    """
    [[1 2 3]
    [4 5 6]]
    """

    # ravel方法将多维数组展平成一维数组
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    flattened = matrix.ravel()
    print("ravel:二维数组转一维数组", flattened)  # [1 2 3 4 5 6]

    matrix = np.array([[[1, 2, 3], [4, 5, 6]], [[11, 22, 33], [44, 55, 66]]])
    flattened = matrix.ravel()
    print("ravel:三维数组转一维数组", flattened)  # [ 1  2  3  4  5  6 11 22 33 44 55 66]

    # flatten把多维数组转换为⼀维数组，注意每个元组的⻓度是相同的
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    flattened = matrix.flatten()
    print("flatten:二维数组转一维数组", flattened)  # [1 2 3 4 5 6]

    matrix = np.array([[[1, 2, 3], [4, 5, 6]], [[11, 22, 33], [44, 55, 66]]])
    flattened = matrix.flatten()
    print("flatten:三维数组转一维数组", flattened)  # [ 1  2  3  4  5  6 11 22 33 44 55 66]

    # transpose方法用于矩阵的转置操作，交换数组的维度
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    # 转置 darray.T
    print(matrix.T)
    """
    [[1 4]
    [2 5]
    [3 6]]
    """
    transposed = matrix.transpose()
    print(transposed)
    """
    [[1 4]
    [2 5]
    [3 6]]
    """

    # newaxis : 增加维度
    newaxis_arr = np.array([1, 2, 3])
    print(newaxis_arr.shape)  # (3,)

    newaxis_arr = newaxis_arr[:, np.newaxis]
    print(newaxis_arr)
    """
    [[1]
    [2]
    [3]]
    """
    print(newaxis_arr.shape)  # (3, 1)


def arithmetic_operations():
    # 数组的运算
    # 算数运算
    arr_1 = np.array([1, 2, 3])
    arr_2 = np.array([1, 2, 3])

    print(arr_1 + arr_2)  # [2 4 6]
    print(arr_1 - arr_2)  # [0 0 0]
    print(arr_1 * arr_2)  # [1 4 9]
    print(arr_1 / arr_2)  # [1. 1. 1.]
    print("---")

    # 求和、求积
    arr_3 = np.array([1, 2, 6])
    print("sum 求和：", arr_3.sum())
    print("sum 求积：", arr_3.prod())
    print("---")

    # 平均数，⽅差，标准差，最⼤值，最⼩值
    arr_4 = np.array([1, 3, 5])
    print("mean 平均数:", arr_4.mean())
    print("var 方差:", arr_4.var())
    print("std 标准差:", arr_4.std())
    print("max 最大值:", arr_4.max())
    print("min 最⼩值:", arr_4.min())
    print("---")

    # 最⼤与最⼩值对应的索引值：argmax,argmin:
    # 取元素值上限，下限，四舍五⼊：ceil, floor, rint
    arr_5 = np.array([1.2, 3.8, 4.9])
    print("argmax 最大值索引:", arr_5.argmax())
    print("argmin 最小值索引:", arr_5.argmin())
    print("ceil 取元素值上限:", np.ceil(arr_5))
    print("floor 取元素值下限:", np.floor(arr_5))
    print("rint 四舍五入:", np.rint(arr_5))
    print("---")

    # 标量运算
    print(arr_1 * 10)  # [10 20 30]
    print("---")

    # 广播是NumPy的一个强大特性，它允许对形状不同的数组进行算术运算。
    # NumPy会自动扩展较小的数组，使得它们的形状兼容，从而完成运算
    arr_1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr_2 = np.array([1, 0, 1])

    print(arr_1 + arr_2)
    """
    [[2 2 4]
    [5 5 7]]
    """


def matrix_operations():
    # Numpy矩阵操作（高级）
    # 矩阵乘法
    arr_1 = np.array([[1, 2], [3, 4]])
    arr_2 = np.array([[5, 6], [7, 8]])

    # 使用dot函数进行矩阵乘法(arr_1 横向行项分别跟arr_2纵向分别相乘的和)
    # 使用np.dot()函数进行了矩阵乘法，结果是两个矩阵的标准矩阵乘积
    result = np.dot(arr_1, arr_2)
    print(result)
    """
    arr_1=a
    arr_2=b

    a11 * b11 + a12 * b21 = 1*5+2*7=19
    a11 * b12 + a12 * b22 = 1*6+2*8=22
    a21 * b11 + a22 * b21 = 3*5+4*7=43
    a21 * b12 + a22 * b22 = 3*6+4*8=50

    [[19 22]
    [43 50]]
    """
    # dot 等同于 @
    print(arr_1 @ arr_2)

    # * 对比dot
    result = arr_1 * arr_2
    print(result)
    """
    [[ 5 12]
    [21 32]]
    """

    # 矩阵的逆，使用np.linalg.inv()函数来计算矩阵的逆
    arr = np.array([[1, 2], [3, 4]])
    arr_inv = np.linalg.inv(arr)
    print(arr_inv)

    """
    [[-2.   1. ]
    [ 1.5 -0.5]]
    """

    arr = np.array([[-2., 1.], [1.5, -0.5]])
    arr_inv = np.linalg.inv(arr)
    print(arr_inv)
    """
    [[1. 2.]
    [3. 4.]]
    """

    # arr = np.array([[0, 0], [3, 4]])
    # arr_inv = np.linalg.inv(arr)
    # print(arr_inv) # LinAlgError: Singular matrix


def broadcast():
    # 广播机制
    arr_1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr_2 = np.array([1, 0, 1])  # 缺失的维度，即[[1,0,1],[1,0,1]]

    result = arr_1 + arr_2
    print(result)
    """
    [[2 2 4]
    [5 5 7]]
    """

    arr_1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr_2 = np.array([[1, 0, 1], [3, 3, 3]])

    result = arr_1 + arr_2
    print(result)
    """
    [[2 2 4]
    [7 8 9]]
    """

    arr_1 = np.array([[1, 2, 3],
                      [4, 5, 6]])  # shape (2, 3)

    arr_2 = np.array([10, 20, 30])  # shape (3,)

    result = arr_1 + arr_2
    """
    1. 首先，NumPy 会将arr_2 的形状从 (3,) 扩展为 (1, 3)
    2. 然后，NumPy 会将arr_2沿着第一个维度复制，使其形状变为(2, 3)
    3. 最后，进行元素级的加法操作。
    arr_1 + arr_2 = [[1+10, 2+20, 3+30],
                    [4+10, 5+20, 6+30]] 
                = [[11, 22, 33],
                    [14, 25, 36]]
    """

    # 广播机制的限制,两个数组在某个维度上都不为1，且大小不相同，NumPy 会抛出 `ValueError` 异常，表示无法进行广播
    try:
        arr_1 = np.array([[1, 2, 3],
                          [4, 5, 6]])  # shape (2, 3)

        arr_2 = np.array([10, 20])  # shape (2,)
        result = arr_1 + arr_2
    except Exception as ex:
        print(ex)

    # 随机数生成
    # ⽣成指定⻓度，在 [0,1) 之间平均分布的随机数组：
    random_arr = np.random.random(5)
    print(random_arr)
    print("---")

    # 生成一个3x3的随机数组，元素在[0, 1)之间
    rand_arr = np.random.rand(3, 3)
    print(rand_arr)
    print("---")

    # ⽣成指定⻓度，符合正态分布的随机数组，指定其均值为 0，标准差为 0.1：
    mu, sigma = 0, 0.1
    normal_arr = np.random.normal(mu, sigma, 5)
    print(normal_arr)
    print("---")

    # 生成一个服从标准正态分布的随机数组
    normal_arr = np.random.randn(3, 3)
    print(normal_arr)
    print("---")

    # 生成一个0到10之间的随机整数数组
    int_arr = np.random.randint(0, 10, size=(3, 3))
    print(int_arr)


if __name__ == '__main__':
    numpy_base()

    inner_func()

    const_func()

    index_and_slice()

    numpy_shape()

    arithmetic_operations()

    matrix_operations()

    broadcast()
