from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 加载鸢尾花数据集
X, y = load_iris(return_X_y=True)
# 只取前100个样本（只考虑类别0和类别1）
X = X[:100]
y = y[:100]  # 这样y只包含0和1两个类别

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"类别分布: {np.bincount(y)}")


# 2. 定义逻辑回归模型
class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.bias = None
        self.training_history = {'loss': [], 'accuracy': []}

    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """前向传播计算"""
        # 线性运算: z = X * theta + bias
        z = np.dot(X, self.theta.T) + self.bias  # X形状(m,n_features)，theta.T形状(n_features,1)
        # 应用sigmoid函数
        y_hat = self.sigmoid(z)
        return y_hat

    def compute_loss(self, y, y_hat):
        """计算二元交叉熵损失"""
        epsilon = 1e-8  # 防止log(0)
        # 二元交叉熵损失
        loss = -y * np.log(y_hat + epsilon) - (1 - y) * np.log(1 - y_hat + epsilon)
        return np.mean(loss)

    def compute_gradients(self, X, y, y_hat):
        """计算参数梯度"""
        m = X.shape[0]
        # 计算误差 (确保形状一致)
        error = y_hat.flatten() - y  # 确保error是一维数组，形状(m,)

        # 计算梯度
        # 正确处理矩阵乘法，确保形状一致
        dtheta = np.zeros_like(self.theta)  # 初始化与theta相同形状
        for i in range(m):
            dtheta += error[i] * X[i, :]
        dtheta = dtheta / m

        # 计算偏置梯度
        dbias = np.mean(error)

        return dtheta, dbias

    def fit(self, X, y, verbose=True):
        """训练模型"""
        # 初始化参数
        num_features = X.shape[1]
        self.theta = np.random.randn(1, num_features) * 0.01  # 小的随机初始值
        self.bias = 0

        # 训练过程
        for epoch in range(self.epochs):
            # 前向传播
            y_hat = self.forward(X)  # 形状应为(m,1)或(m,)

            # 计算损失
            loss = self.compute_loss(y, y_hat)
            self.training_history['loss'].append(loss)

            # 计算准确率 (确保形状一致)
            predictions = (y_hat.flatten() >= 0.5).astype(int)  # 确保是一维数组
            accuracy = np.mean(predictions == y)
            self.training_history['accuracy'].append(accuracy)

            # 计算梯度
            dtheta, dbias = self.compute_gradients(X, y, y_hat)

            # 更新参数
            self.theta -= self.learning_rate * dtheta
            self.bias -= self.learning_rate * dbias

            # 打印训练进度
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}: loss = {loss:.6f}, accuracy = {accuracy:.4f}")

    def predict(self, X):
        """预测类别"""
        y_hat = self.forward(X)
        predictions = (y_hat >= 0.5).astype(int)
        return predictions

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)

    def save_model(self, filename):
        """保存模型参数"""
        model_params = {
            'theta': self.theta,
            'bias': self.bias
        }
        try:
            np.savez(filename, **model_params)
            print(f"模型参数已保存至 {filename}")
        except Exception as e:
            print(f"保存模型参数时出错: {e}")
            # 尝试保存到临时目录
            try:
                import os
                import tempfile

                temp_dir = tempfile.gettempdir()
                filepath = os.path.join(temp_dir, 'iris_model.npz')
                np.savez(filepath, **model_params)
                print(f"模型参数已保存至 {filepath}")
            except Exception as e:
                print(f"无法保存模型参数: {e}")


# 3. 实验不同的超参数组合
def run_experiment(learning_rates, test_sizes):
    """运行实验，尝试不同的学习率和测试集比例"""
    results = {}

    for test_size in test_sizes:
        for lr in learning_rates:
            print(f"\n实验: 学习率 = {lr}, 测试集比例 = {test_size}")

            # 数据拆分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # 创建并训练模型
            model = LogisticRegression(learning_rate=lr, epochs=1000)
            model.fit(X_train, y_train, verbose=False)

            # 评估模型
            train_accuracy = np.mean(model.predict(X_train) == y_train)
            test_accuracy = np.mean(model.predict(X_test) == y_test)

            # 存储结果
            key = f"lr={lr}, test_size={test_size}"
            results[key] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'params': {'learning_rate': lr, 'test_size': test_size}
            }

            print(f"训练集准确率: {train_accuracy:.4f}")
            print(f"测试集准确率: {test_accuracy:.4f}")

    return results

if __name__ == "__main__":
    # 4. 运行实验
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    test_sizes = [0.2, 0.3, 0.4]

    results = run_experiment(learning_rates, test_sizes)

    # 5. 找出最佳模型
    best_model_key = max(results, key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_model_key]['model']
    print(f"\n最佳模型: {best_model_key}")
    print(f"测试集准确率: {results[best_model_key]['test_accuracy']:.4f}")

    # 6. 保存最佳模型
    best_model.save_model('best_iris_model.npz')

    # 7. 对测试样本进行简单预测示例
    idx = np.random.randint(len(X))
    sample = X[idx]
    true_label = y[idx]
    pred = best_model.predict(sample.reshape(1, -1))[0]
    prob = best_model.predict_proba(sample.reshape(1, -1))[0]

    print(f"\n预测示例:")
    print(f"样本特征: {sample}")
    print(f"真实标签: {true_label}")
    print(f"预测标签: {pred}")