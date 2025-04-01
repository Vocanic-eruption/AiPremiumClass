#【第四周作业】

# 1. 搭建的神经网络，使用olivettiface数据集进行训练。
# 2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
# 3. 尝试不同optimizer对模型进行训练，观察对比loss结果。
# 4. 无需提交：注册kaggle并尝试激活Accelerator，使用GPU加速模型训练。

# 1.引入olivettiface数据集
from sklearn.datasets import fetch_olivetti_faces

# 2.加载数据集
olivetti_faces = fetch_olivetti_faces()
# print(dir(olivetti_faces))
# print(olivetti_faces.DESCR)
# data = olivetti_faces.data
# images = olivetti_faces.images
# target = olivetti_faces.target
# print(target)

# 3.初始化特征和标签
x = olivetti_faces.data
y = olivetti_faces.target
# 此时，X和Y里面已经被装满了400张（每张都是64*64像素的数据描述）照片的描述和特征

# 4.区分数据集和训练集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 5.初始化模型，使用SVC（支持向量机）模型
from sklearn.svm import SVC
model = SVC()
# 使用该模型原因如下：
# SVM 在处理具有清晰边界的分类问题时表现出色，能够找到一个最优的超平面来分隔不同类别的数据。
# olivetti_faces 数据集相对来说样本数量有限（400 张图像），SVM 在小样本数据集上通常能有较好的表现，不易出现过拟合现象
# SVM 可以通过核函数将原始数据映射到更高维的特征空间，从而更好地处理非线性可分的数据。
# SVM 具有较好的泛化能力，即能够在新的、未见过的数据上表现出较好的分类性能。

# 6.模型训练
model.fit(x_train,y_train)

# 7.模型预测
y_pred = model.predict(x_test)

# 8.模型评估
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")
# 测试了一下，准确率为92.50%

# 9. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
# 归一化处理
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)

# 正则化SVC模型
model = SVC(C=0.1)
# 对C参数进行正则化,尝试【0.01， 0.1，1，10】对结果的影响

#训练模型
model.fit(x_train_scale , y_train)

# 测试模型
y_pred = model.predict(x_test_scale)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")









