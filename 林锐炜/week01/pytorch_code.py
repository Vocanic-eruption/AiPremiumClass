import torch
import numpy as np


def tensor_base():
    # 张量创建及基础属性
    # mac m芯片设置device mps
    # tensor_data = torch.tensor([[1,2],[3,4]], dtype=torch.float32,device="mps:0")
    tensor_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    print("张量：", tensor_data)
    print("形状shape：", tensor_data.shape)
    print("维度ndim：", tensor_data.ndim)
    print("张量元素类型：", tensor_data.dtype)
    print("张量类型：", type(tensor_data))
    print("张量的大小：", tensor_data.size())

    # 检测是否支持GPU
    print(torch.cuda.is_available())
    print("张量 device：", tensor_data.device)

    # 设置张量在GPU上运算
    if torch.cuda.is_available():
        tensor = tensor_data.to('cuda')


def gpu_check():
    # mac m1
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 英伟达GPU
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


def inner_func():
    # torch.ones
    tensor_data = torch.ones(3, 3, dtype=torch.int32)
    print("tensor_data", tensor_data)

    # torch.zeros
    zeros_data = torch.zeros(3, 4, dtype=torch.float32)
    print("zeros_data:", zeros_data)

    # zero_like
    rand_data = torch.rand(4, 3)
    print("rand_data:", rand_data)
    zeros_like_data = torch.zeros_like(rand_data)
    print("zeros_like_data:", zeros_like_data)

    # ones_like
    ones_like_data = torch.ones_like(rand_data, dtype=torch.int32)
    print("ones_like:", ones_like_data)

    # 使用张量维度元组
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    print("rand_tensor:", {rand_tensor})
    ones_tensor = torch.ones(shape)
    print("ones_tensor:", {ones_tensor})
    zeros_tensor = torch.zeros(shape)
    print("zeros_tensor:", {zeros_tensor})

    # 均匀分布
    print(torch.rand(6, 5))
    # 标准正态分布
    print(torch.randn(6, 5))
    # 离散正态分布
    print(torch.normal(mean=.0, std=1.0, size=(6, 5)))
    # 线性间隔向量(返回⼀个1维张量，包含在区间start和end上均匀间隔的steps个点)
    print(torch.linspace(start=1, end=10, steps=10, dtype=torch.float32))

    tensor_1 = torch.tensor([1, 2, 3])
    print(tensor_1, "\n[1:]:", tensor_1[1:])

    tensor_2 = torch.tensor([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    print(tensor_2, "\n[1]:", tensor_2[0])
    print(tensor_2, "\n[1][1]:", tensor_2[1][1])
    print(tensor_2, "\n[1,][1,]:", tensor_2[..., 2])


def tensor_concat():
    # torch.cat 横向拼接
    tensor_1 = torch.tensor([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    t1 = torch.cat([tensor_1, tensor_1, tensor_1], dim=1)
    print(t1)

    tensor_2 = torch.tensor([1, 2, 3])
    t2 = torch.cat([tensor_2, tensor_2, tensor_2])
    print(t2)

    # torch.stack 纵向拼接
    tensor_1 = torch.tensor([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    t1 = torch.stack([tensor_1, tensor_1, tensor_1], dim=2)
    print(t1)

    tensor_2 = torch.tensor([1, 2, 3])
    t2 = torch.stack([tensor_2, tensor_2, tensor_2])
    print(t2)


def arithmetic_operations():
    tensor_1 = torch.tensor([(1, 2, 3), (4, 5, 6)])
    tensor_2 = torch.tensor([(1, 2, 3), (4, 5, 6)])

    result = tensor_1 + tensor_2
    print("张量相加+:", result)
    print("张量相加add:", tensor_1.add(tensor_2))
    """
 tensor([[ 2,  4,  6],
         [ 8, 10, 12]])
 """
    print("---")

    result = tensor_1 - tensor_2
    print("张量相减-:", result)
    print("张量相减sub:", tensor_1.sub(tensor_2))
    """
 tensor([[0, 0, 0],
         [0, 0, 0]])

 """
    print("---")

    result = tensor_1 * tensor_2
    print("张量相乘×:", result)
    print("张量相乘mul:", tensor_1.mul(tensor_2))
    """
 tensor([[1, 4, 9],
         [16, 25, 36]])
 """
    print("---")

    result = tensor_1 / tensor_2
    print("张量相除÷:", result)
    print("张量相除div:", tensor_1.div(tensor_2))
    """
 tensor([[1., 1., 1.],
         [1., 1., 1.]])
 """
    print("---")
    # 张量乘积
    tensor_1 = torch.tensor([(1, 2), (3, 4)])
    tensor_2 = torch.tensor([(5, 6), (7, 8)])
    #  mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
    result = tensor_1 @ tensor_2

    """
 tensor_1=a
 tensor_2=b

 a11 * b11 + a12 * b21 = 1*5+2*7=19
 a11 * b12 + a12 * b22 = 1*6+2*8=22
 a21 * b11 + a22 * b21 = 3*5+4*7=43
 a21 * b12 + a22 * b22 = 3*6+4*8=50

 [[19 22]
  [43 50]]
 """

    print(result)
    print("matmul:", tensor_1.matmul(tensor_2))
    print("---")
    print("tensor_2.T", tensor_2.T)
    print("matmul:", tensor_1.matmul(tensor_2.T))


def matmul_operations():
    # Shape = (n, n)
    tensor_a = torch.randn(3, 3)  # 形状 (3, 3)
    tensor_b = torch.randn(3, 3)  # 形状 (3, 3)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 (3, 3)
    print("Shape = (n, n)\n", tensor_c)

    # 一维向量点积 (n,) 和 (n,)->()
    tensor_a = torch.randn(4)  # 形状 (4,)
    tensor_b = torch.randn(4)  # 形状 (4,)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 ()
    print("(n,) 和 (n,)->()\n", tensor_c)

    # 广播机制(b, n, m) 和 (m, p)->(b, n, p)
    tensor_a = torch.randn(2, 3, 4)  # 形状 (2, 3, 4)
    tensor_b = torch.randn(4, 5)  # 形状 (4, 5)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 (2, 3, 5)
    print("(b, n, m) 和 (m, p)->(b, n, p)\n", tensor_c)

    tensor_a = torch.randn(2, 3, 4)  # 形状 (2, 3, 4)
    tensor_b = torch.randn(4, 4)  # 形状 (2, 4)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 (2, 3)
    print(tensor_c)

    # 矩阵与向量相乘(n, m) 和 (m,)->(n,)
    tensor_a = torch.randn(3, 4)  # 形状 (3, 4)
    tensor_b = torch.randn(4)  # 形状 (4,)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 (3,)
    print("(n, m) 和 (m,)->(n,)\n", tensor_c)

    # (b, n, m) 和 (b, m, p)->(b, n, p)
    tensor_a = torch.randn(2, 3, 4)  # 形状 (2, 3, 4)
    tensor_b = torch.randn(2, 4, 5)  # 形状 (2, 4, 5)
    tensor_c = torch.matmul(tensor_a, tensor_b)  # 形状 (2, 3, 5)
    print("(b, n, m) 和 (b, m, p)->(b, n, p)\n", tensor_c)

    tensor_a = torch.tensor([[1], [2], [3]])  # 形状(1,3)
    tensor_b = torch.tensor([10, 20])  # 形状(2,2)
    print(tensor_a + tensor_b)


def tensor_numpy_switch():
    # tensor to ndarray
    tensor = torch.ones(5)
    print(tensor)
    arr = tensor.numpy()
    print("arr:", arr)
    print("arr type:", type(arr))

    # ndarray to tensor
    arr = np.ones(5)
    print(arr)
    tensor = torch.from_numpy(arr)
    print("tensor:", tensor)
    print("tensor type:", type(tensor))

    # torch.from_numpy 张量可以从 NumPy 数组创建
    arr = np.array([(123, 456), (1, 2)], dtype=float)
    tensor_data = torch.from_numpy(arr)
    print(tensor_data)


if __name__ == '__main__':
    tensor_base()

    gpu_check()

    inner_func()

    tensor_concat()

    arithmetic_operations()

    matmul_operations()

    tensor_numpy_switch()
