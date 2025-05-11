import numpy as np
import tensorflow as tf
import os
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


"""
1. 实验使用不同的RNN结构,实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
"""
# 数据预处理,划分数据集
def load_data(test_size,random_state):
    data = fetch_olivetti_faces()
    images = data.images.reshape(data.images.shape[0], data.images.shape[1], data.images.shape[2], -1)  # (400, 4096)
    labels = data.target
    
    # 数据标准化
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    
    # 转换为RNN输入格式：(样本数, 时间步长, 特征维度)
    # 将4096像素视为时间步长=4096，每个时间步特征维度=1
    X = images.reshape(-1, 4096, 1)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state
    )
    
    # 标签编码
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    
    return X_train, X_test, y_train, y_test

# 模型构建
def build_model(model_type):
    model = tf.keras.Sequential()
    # 通用RNN结构
    if model_type == "RNN":
        model.add(tf.keras.layers.SimpleRNN(128, input_shape=(4096, 1)))
    elif model_type == "LSTM":
        model.add(tf.keras.layers.LSTM(128, input_shape=(4096, 1)))
    elif model_type == "GRU":
        model.add(tf.keras.layers.GRU(128, input_shape=(4096, 1)))
    elif model_type == "BiRNN":
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(64), 
            input_shape=(4096, 1)
        ))
    
    # 分类层
    model.add(tf.keras.layers.Dense(40, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

def buildlogpath(log_dir):
    # 确保父目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)  # 递归创建目录‌:ml-citation{ref="1,5" data="citationList"}


def train_models():
    # 统一训练参数
    epochs = 30
    batch_size = 16
    test_size=0.2
    random_state=42
    X_train, X_test, y_train, y_test = load_data(test_size,random_state)

    models = {
        "RNN": build_model("RNN"),
        "LSTM": build_model("LSTM"),
        "GRU": build_model("GRU"),
        "BiRNN": build_model("BiRNN")
    }

    for name, model in models.items():
        log_dir = f"D://logs/{name}"
        buildlogpath(log_dir)
        # 配置 TensorBoard 回调
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

        print(f"\nTraining {name}...")

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tensorboard_callback]
        )
        
        # 最终评估
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name} Test Accuracy: {acc:.4f}; Test Loss: {loss:.4f}")


if __name__ == "__main__":
    train_models()
