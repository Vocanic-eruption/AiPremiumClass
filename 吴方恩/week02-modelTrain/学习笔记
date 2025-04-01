# NumPy 与 PyTorch 核心概念与操作对比笔记

---

## 一、数据容器基础
### 1.1 核心数据结构
| **特性**          | **NumPy (ndarray)**                          | **PyTorch (Tensor)**                          |
|-------------------|---------------------------------------------|-----------------------------------------------|
| **创建方式**       | `np.array([1,2,3])`                         | `torch.tensor([1,2,3])`                       |
| **特殊矩阵**       | `np.zeros()`, `np.ones()`, `np.eye()`       | `torch.zeros()`, `torch.ones()`, `torch.eye()` |
| **随机生成**       | `np.random.random()`, `np.random.normal()`  | `torch.rand()`, `torch.randn()`               |
| **序列生成**       | `np.arange()`, `np.linspace()`              | `torch.arange()`, `torch.linspace()`          |

**代码示例**（文档1）：
```python
# NumPy 特殊矩阵
a1 = np.zeros((2,3), dtype=np.float32)  # 全零矩阵
a3 = np.eye(3)  # 单位矩阵
a4 = np.random.random(5)  # 均匀分布随机数
```

**代码示例**（文档2）：
```python
# PyTorch 张量创建
data = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
rand_tensor = torch.rand(2,3)  # 均匀分布
normal_tensor = torch.randn(2,3)  # 标准正态分布
```

---

## 二、容器属性与元数据
### 2.1 关键属性对比
| **属性**          | **NumPy**              | **PyTorch**            |
|-------------------|------------------------|------------------------|
| 维度数量          | `.ndim`                | `.dim()`               |
| 形状              | `.shape`               | `.shape` / `.size()`   |
| 元素总数          | `.size`                | `.numel()`             |
| 数据类型          | `.dtype`               | `.dtype`               |
| 存储设备          | 仅限CPU                | `.device` (CPU/GPU)    |

**代码示例**（文档1）：
```python
a = np.array([(1,2,3), (4,5,6)])
print(a.ndim)   # 2
print(a.shape)  # (2,3)
print(a.dtype)  # int64
```

**代码示例**（文档2）：
```python
tensor = torch.rand(3,4)
print(tensor.shape)   # torch.Size([3,4])
print(tensor.device)  # cpu / cuda:0
```

---

## 三、数学运算体系
### 3.1 基本运算对比
| **运算类型**       | **NumPy**               | **PyTorch**             |
|--------------------|-------------------------|-------------------------|
| 逐元素运算         | `+`, `-`, `*`, `/`      | `+`, `-`, `*`, `/`      |
| 矩阵乘法           | `np.dot(a,b)` / `a @ b` | `torch.matmul(a,b)`     |
| 统计运算           | `.sum()`, `.mean()`     | `.sum()`, `.mean()`     |
| 聚合运算           | `.max()`, `.argmax()`   | `.max()`, `.argmax()`   |

**代码示例**（文档1）：
```python
# 矩阵乘法
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
result_dot = np.dot(a, b)
result_at = a @ b
```

**代码示例**（文档2）：
```python
# 张量运算
y1 = tensor @ tensor.T  # 矩阵乘法
z1 = tensor * tensor    # 逐元素相乘
agg = tensor.sum()      # 聚合求和
```

---

## 四、维度操作与变形
### 4.1 常用变形方法
| **操作**           | **NumPy**               | **PyTorch**             |
|--------------------|-------------------------|-------------------------|
| 改变形状           | `.reshape()`            | `.reshape()`            |
| 转置               | `.T`                    | `.T` / `.t()`           |
| 维度扩展           | `np.newaxis`            | `.unsqueeze()`          |
| 展平               | `.flatten()`            | `.flatten()`            |

**代码示例**（文档1）：
```python
a7 = np.arange(1,10).reshape(3,3,1)  # 三维变形
a = a.T  # 转置操作
```

**代码示例**（文档2）：
```python
t1 = torch.cat([tensor, tensor], dim=1)  # 维度拼接
tensor = tensor.unsqueeze(1)  # 增加维度
```

---

## 五、设备管理与加速
### 5.1 PyTorch 设备操作
```python
# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)

# Mac M系列芯片支持
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)
```

---

## 六、自动微分与计算图
### 6.1 PyTorch 自动求导
```python
# 定义可训练参数
A = torch.randn(10, 10, requires_grad=True)
b = torch.randn(10, requires_grad=True)

# 前向计算
result = torch.matmul(A, x) + torch.dot(b, x)

# 反向传播
result.backward()
print(A.grad)  # 查看梯度
```

### 6.2 计算图可视化
```python
from torchviz import make_dot
dot = make_dot(result, params={'A': A, 'b': b})
dot.render('graph', format='png')  # 生成计算图
```
> 注：需安装Graphviz并配置环境变量

---

## 七、文件与数据交互
### 7.1 数据持久化
| **操作**           | **NumPy**               | **PyTorch**             |
|--------------------|-------------------------|-------------------------|
| 保存数据           | `np.save('arr.npy', a)` | `torch.save(tensor, 'tensor.pt')` |
| 加载数据           | `np.load('arr.npy')`    | `torch.load('tensor.pt')` |

**代码示例**（文档1）：
```python
np.save('result.npy', manual_result)
loaded = np.load('result.npy')
```

---

## 八、最佳实践要点
1. **内存共享**：PyTorch的`from_numpy()`与NumPy数组共享内存
2. **类型转换**：注意`torch.float32`与`numpy.float32`的对应关系
3. **设备同步**：GPU张量需先转CPU才能转换为NumPy数组
4. **梯度控制**：`with torch.no_grad()`可禁用梯度跟踪提升性能

```python
# 内存共享示例
numpy_array = np.ones(5)
torch_tensor = torch.from_numpy(numpy_array)  # 共享内存
numpy_array[0] = 2  # torch_tensor也会同步修改
```