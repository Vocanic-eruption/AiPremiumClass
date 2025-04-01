from sklearn.model_selection import train_test_split
import numpy as np
#use iris dataset sample
from sklearn.datasets import load_iris

# 1.数据准备，参数初始化
X,y  = load_iris(return_X_y=True)
#取前100个数据
X[:100]
#取前100个标签（0,1）
y[:100]

#拆分训练和测试集,可以修改test_size参数调整测试集比例
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, shuffle=True)

#初始化参数模型
theta = np.random.randn(1,4)
bias = 0
#学习率，基于经验，最近学习率在0.1-0.01之间，此处可以通过修改lr参数调整学习率
lr = 0.01
#模型训练的轮次
epochs = 1000

# 2.模型运算（模型训练）
def forward(x, theta, bias):
    # linear 
    z = np.dot(theta,x.T) + bias 
    # sigmoid 
    y_hat = 1 / (1 + np.exp(-z)) 
    return y_hat

# 3.计算损失（损失函数）
def loss_function(y, y_hat):
    e = 1e-8 # 防止 y_hat计算值为0，添加的极小 值epsilon
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 4.计算梯度
def calc_gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_theta = np.dot(y_hat-y,x)/m 
    delta_bias = np.mean(y_hat-y) 
    return delta_theta, delta_bias


for i in range(epochs):
    #前向传播
    y_hat = forward(train_X,theta,bias)
    #计算损失
    loss = np.mean(loss_function(train_y,y_hat))
    if i % 100 == 0:
        print('step:',i,'loss:',loss)
    #梯度下降
    dw,db = calc_gradient(train_X,train_y,y_hat)
    #更新参数
    theta -= lr * dw 
    bias -= lr * db

#这里可以报错训练模型
# 保存训练好的参数
np.savez('model_params.npz', theta=theta, bias=bias)
print('模型参数已保存到 model_params.npz')

#这里可以加载之前保存的训练参数theta, bias
# 加载保存的参数
loaded_params = np.load('model_params.npz')
theta = loaded_params['theta']
bias = loaded_params['bias']


# 5.模型测试
# 测试模型
idx = np.random.randint(len(test_X))
x = test_X[idx]
y = test_y[idx]

def predict(x): 
    pred = forward(x,theta,bias)[0] 
    if pred > 0.5:
        return 1 
    else: 
        return 0
    
pred = predict(x)
print(f'预测值：{pred} 真实值：{y}')
