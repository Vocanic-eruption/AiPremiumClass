from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 生成数据集
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 2. 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 3. 初始化模型参数
theta = np.random.randn(1, X.shape[1])  # 权重参数 (1, n_features)
bias = 0                                # 截距项
lr = 0.5                                # 初始学习率
epochs = 2000                           # 训练轮数

# 4. 前向传播函数（修正维度问题）
def forward(x, theta, bias):
    z = np.dot(theta, x.T) + bias       # (1, n_samples)
    y_hat = 1 / (1 + np.exp(-z))        # Sigmoid激活
    return y_hat.squeeze()              # 输出形状 (n_samples,)

# 5. 交叉熵损失函数
def compute_loss(y, y_hat):
    epsilon = 1e-8
    return -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))

# 6. 梯度计算函数
def calc_gradient(x, y, y_hat):
    m = x.shape[0]
    delta_theta = np.dot((y_hat - y), x) / m  # 权重梯度
    delta_bias = np.mean(y_hat - y)           # 截距梯度
    return delta_theta, delta_bias

# 7. 模型训练
loss_history = []
for epoch in range(epochs):
    y_hat = forward(X_train, theta, bias)
    loss = compute_loss(y_train, y_hat)
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat)
    
    # 更新参数
    theta -= lr * delta_theta
    bias -= lr * delta_bias
    
    # 记录损失
    loss_history.append(loss)
    
    if epoch % 100 == 0:
        acc = np.mean(np.round(y_hat) == y_train)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}")

np.save("theta.npy", theta)
np.save("bias.npy", bias)
np.save("X_test.npy",X_test)
np.save("Y_test.npy",Y_test)

theta = np.load("theta.npy")
bias = np.load("bias.npy")

def forward(x, theta, bias):
    z = np.dot(theta,x.T)+bias
    y_hat=1/(1+np.exp(-z))
    return y_hat

idx = np.random.randint(len(X_test))
x = x_test[idx]
y= y_test[idx]
predict=np.round(forward(x, theta, bias))
print(f"y: {y}, 预测结果: {predict}")
