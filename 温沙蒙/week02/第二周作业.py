from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_classification
# ⽣成观测值数据
X, y = make_classification(n_features=10)
print(X)
print(y)
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)

# 初始化参数模型参数
theta = np.random.randn(1, 10)
print(theta)
bias=0
#学习率
lr=0.001
#迭代次数
epoch=5000

#计算函数
def forward(X):
    z=np.dot(theta,X.T) + bias
    y_hat=1/(1+np.exp(-z))
    return y_hat

#损失函数
def loss(y,y_hat):
    e=1e-5
    return -np.mean(y*np.log(y_hat+e)+(1-y)*np.log(1-y_hat+e))

#计算梯度
def gradient(X,y,y_hat):
    m=X.shape[-1]
    dz=y_hat-y
    dw=np.dot(dz,X)/m
    db=np.sum(dz)/m
    return dw,db

#模型训练
for i in range(epoch):
    y_hat=forward(X_train)
    l=loss(y_train,y_hat)
    dw,db=gradient(X_train,y_train,y_hat)
    theta=theta-lr*dw
    bias=bias-lr*db
    if i%100==0:
        print(f'epoch:{i},loss:{l}')    

    #计算准确率
    acc=(y_hat>0.5)==y_train
    print("epoch:",i,"loss:",np.mean(l),"acc:",np.mean(acc))

print("theta:",theta)
print("bias:",bias)
#输出参数到文件
np.savetxt("theta.txt",theta)
np.savetxt("bias.txt",np.array([bias]))

#读取参数
theta=np.loadtxt("theta.txt")
bias=np.loadtxt("bias.txt")
           