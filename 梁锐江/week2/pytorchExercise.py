import torch
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

X, y = make_classification(n_features=10)

tensor_x = torch.tensor(X, dtype=torch.float)
tensor_y = torch.tensor(y, dtype=torch.float)

w = torch.randn(1, 10, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learn_rate = 1e-3

if __name__ == '__main__':
    for i in range(10000):
        # 前向计算
        r = torch.nn.functional.linear(tensor_x, w, b)
        r = torch.nn.functional.sigmoid(r)

        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy(r.squeeze(1), tensor_y, reduction='mean')

        # 计算梯度
        loss.backward()

        # 参数更新
        with torch.autograd.no_grad():  # 关闭梯度计算跟踪
            w.data = w - learn_rate * w.grad
            # w -= learn_rate * w.grad
            w.grad.zero_()
            b.data = b - learn_rate * b.grad
            b.grad.zero_()

        if i % 1000 == 0:  # 减少打印频率以简化输出
            print(f'train loss: {loss.item():.4f} ')
