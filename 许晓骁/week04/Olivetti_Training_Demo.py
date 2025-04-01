import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Densye, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical


# 加载 Olivetti 数据集
data = fetch_olivetti_faces(shuffle=True, random_state=42)
images = data.images  # 原始图像数据 (400张 64x64 灰度图)
labels = data.target  # 标签 (0-39, 共40个人)

# 显示前5张图片
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()

print("--> 数据加载完成，共有{}张图片，标签范围0-39".format(len(images)))


# 将图像数据转换为4D张量 (样本数, 高度, 宽度, 通道数)
# 灰度图只有1个通道，所以添加最后一个维度1
X = images.reshape(-1, 64, 64, 1).astype('float32') / 255.0  # 归一化到[0,1]

# 将标签转换为One-Hot编码 (例如 3 → [0,0,0,1,0,...])
y = to_categorical(labels)

# 划分训练集和测试集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,  # 保持各类别比例一致
    random_state=42
)

print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)


def build_model():
    model = Sequential(name="Simple_CNN")

    # 第一层：卷积层 + 池化层
    model.add(Conv2D(
        filters=32,  # 32个滤波器（提取32种特征）
        kernel_size=(3, 3),  # 每个滤波器是3x3大小
        activation='relu',  # 激活函数用ReLU（简单且效果好）
        input_shape=(64, 64, 1)  # 输入形状：64x64像素，1个通道（灰度）
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2池化，缩小尺寸

    # 第二层：展平后接全连接层
    model.add(Flatten())  # 将二维特征图转换为一维向量
    model.add(Dense(128, activation='relu'))  # 128个神经元的全连接层

    # 输出层：40个神经元对应40个人
    model.add(Dense(40, activation='softmax'))  # softmax将输出转换为概率

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # 使用Adam优化器
        loss='categorical_crossentropy',  # 分类任务用交叉熵损失
        metrics=['accuracy']  # 监控准确率
    )
    return model


model = build_model()
model.summary()  # 打印模型结构


# 开始训练（epochs表示训练轮数）
history = model.fit(
    X_train, y_train,
    epochs=50,            # 整个数据集训练50次
    batch_size=32,         # 每次用32张图片更新权重
    validation_split=0.2, # 从训练集中取20%作为验证集
    verbose=1              # 显示进度条
)

# 绘制训练曲线
plt.figure(figsize=(12,4))

# 绘制损失曲线
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='训练Loss')
plt.plot(history.history['val_loss'], label='验证Loss')
plt.title('损失曲线')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 在测试集上评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n测试集准确率: {:.2f}%".format(test_acc * 100))

# 随机抽取测试集图片进行预测
sample_idx = np.random.randint(0, len(X_test))
sample_img = X_test[sample_idx]
true_label = np.argmax(y_test[sample_idx])  # 从One-Hot转回数字标签
pred_label = np.argmax(model.predict(sample_img.reshape(1,64,64,1)))

# 显示预测结果
plt.imshow(sample_img.reshape(64,64), cmap='gray')
plt.title(f"真实标签: {true_label}\n预测标签: {pred_label}")
plt.axis('off')
plt.show()
