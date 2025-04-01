# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
x,y = load_iris(return_X_y=True)

x=x[:80]
y=y[:80]

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2)
# 参数设置
theta = np.random.randn(1,4)

eta = 1e-3
bias =0
epochs = 100
# 前向计算
def forward(theta,x,bias):
    # 线性函数
    z = np.dot(theta,x.T) +bias
    # 映射为概率
    y_hat = 1/(1+np.exp(-z))
    return y_hat
# 损失函数
def loss(y,y_hat):
    epsilon = 1e-8
    return -y*np.log(y_hat + epsilon) - (1-y)*np.log(1-y_hat+epsilon)
# 计算梯度
def calu_gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_w =np.dot(y_hat-y,x)/m
    delta_b =np.mean(y_hat-y)
    return delta_w,delta_b
# 模型训练
for i in range(epochs):
    y_hat = forward(theta,train_x,bias)
    loss_1 = np.mean(loss(train_y,y_hat))
    dw,db = calu_gradient(train_x,train_y,y_hat)
    theta -=eta*dw
    bias-=eta*db
# acc的计算
    acc = np.mean(np.round(y_hat) == train_y)
    print(f"i:{i},loss:{np.mean(loss_1)},acc:{acc}")

# %%
np.save('theta.npy',theta)
np.save('bias.npy',bias)

# %%
theta_load = np.load("theta.npy")
bias_load = np.load("bias.npy")
x,y = load_iris(return_X_y=True)
x=x[:100]
y=y[:100]



# %%
x=x[80:]
y=y[80:]

# %%
y_hat = forward(theta_load,x,bias_load)
acc_t = np.mean(np.round(y_hat) == y)
print(acc_t)


