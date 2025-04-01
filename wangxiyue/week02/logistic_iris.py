import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 定义保存模型名称
MODEL_NAME='model.npz'


######################## 作业 ##########################
# 使用sklearn数据集训练逻辑回归模型；
# 调整学习率，样本数据拆分比率，观察训练结果；
# 训练后模型参数保存到文件，在另一个代码中加载参数实现预测功能；
# 总结逻辑回归运算及训练相关知识点。
########################################################

# 逻辑回归步骤（）
#  1 数据准备 （鸢尾花数据集 ， 训练+验证）
### 循环步骤
#  2 模型计算（线性回归函数计算 + sigmoid 线性回归概率分布计算， ）
#  3 损失函数 （ 损失函数结果无线趋近目标，认为是收敛）
#  4 梯度计算 （theta 偏导，bias偏导）
#  5 更新参数
#  6 保存模型
#  7 验证模型

#xTrain, xTest , yTrain,yTest =
def initData():
    #花萼长度、花萼宽度、花瓣长度、花瓣宽度
    x1, y1 = load_iris(return_X_y=True)
    x1=x1[:100]
    y1=y1[:100]
    return train_test_split(x1,y1,test_size=0.2,shuffle=True)

# 模型 得出y的 概率
def calc_model(xTrain,theta,bias):
    #线性运算
    z=np.dot(theta,xTrain.T)+bias
    #sugmoid
    # y_hat = 1 / (1 + np.exp(-z)) # 0，1 概率分布
    #softmax
    y_hat = np.exp(z) / len(z)

    return y_hat

# epsilon : 极小值，防止y_hat 出现 0
def calc_loss(yTrain,y_hat,epsilon):
    #多分类交叉熵损失
    # num_class = yTrain.shape[1]
    # return -np.mean(np.sum(yTrain * np.log(y_hat)))

    return -yTrain * np.log(y_hat+epsilon) - (1-yTrain) * np.log(1-y_hat+epsilon)

# 梯度计算(得出 增量 斜率 和 截距)
def calc_gradient(xTrain,yTrain,y_hat):
    # 梯度
    m = xTrain.shape[-1]
    # theta 梯度
    delta_theta = np.dot((y_hat - yTrain) , xTrain) / m
    # bias 梯度
    delta_bias =  np.mean(y_hat - yTrain)
    return delta_theta, delta_bias


if __name__ == '__main__':


    # 初始化参数
    theta = np.random.randn(1,4)  # shape (1,10)
    bias = 0
    # 学习率
    lr = 0.001
    # 最大训练批次
    epochs = 5000
    #极小值
    epsilon = 1e-8
    xTrain, xTest, yTrain, yTest = initData()

    last_loss = 0
    for i in range(epochs):
        # 向量计算
        y_hat = calc_model(xTrain,theta,bias)
        # 损失计算
        lossVal = calc_loss(yTrain,y_hat,epsilon)
        # 梯度增量
        delta_theta, delta_bias = calc_gradient(xTrain,yTrain,y_hat)
        # 更新参数
        theta = theta - lr * delta_theta
        bias = bias - lr * delta_bias

        # print(theta,bias)
        if i % 10 == 0:
            # 求准确率
            acc = np.mean(np.round(y_hat) == yTrain)
            print(f"epoch:{i} , loss:{np.mean(calc_loss(yTrain, y_hat,epsilon))}  , acc : {acc}")

    np.savez(MODEL_NAME, theta=theta,bias=bias)

def predict_iris(x):
    model_param = np.load(MODEL_NAME)
    theta = model_param['theta']
    bias = model_param['bias']
    return np.round(calc_model(x, theta, bias))[0]