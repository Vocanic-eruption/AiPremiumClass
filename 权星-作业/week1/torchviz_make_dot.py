import torch
from torchviz import make_dot

# 定义矩阵A, 向量 b 和 常数 c
A = torch.randn(10, 10, requires_grad=True)
B = torch.randn(10, requires_grad=True)
C = torch.randn(1, requires_grad=True)
x = torch.randn(10, requires_grad=True)

# 计算函数 f(x) = A * x^T  + B * x + C
result = torch.matmul(A, x.T) + torch.matmul(B, x) + C

# 生成计算图节点
dot = make_dot(result, params={'A': A, 'B': B, 'C': C, 'x': x})
# 绘制计算图
dot.render('expression', format='png', cleanup=True, view=False)


