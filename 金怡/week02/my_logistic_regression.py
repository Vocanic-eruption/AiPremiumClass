import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.verbose = verbose

    def _softmax(self, z):
        '''
        standardize
        z: score-> z-max(z): (-n,0)
        ->exp_z(0,1)
        -> divide by sum 归一化确保总概率和为 1
        '''
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        y_onehot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y):
            idx = np.where(self.classes_ == label)[0][0]
            y_onehot[i, idx] = 1
        # 逻辑回归是凸优化 z=WX+b
        # X(n,d) * W(d,c)->(n,c)+b(1,c)->z(n,c)
        # 初始化为0，z=0 -> softmax=exp0/sum(exp0)=1/c 所有预测概率相等
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        for i in range(self.num_iter):
            logits = np.dot(X, self.weights) + self.bias  # (n_samples, n_classes)
            probs = self._softmax(logits)  # (n_samples, n_classes)
            # p↓ log(p)↑ loss↑   y=[1,0,0] p=[0.8,0.1,0.1] loss小，p=[0,0.5,0.5] loss大
            loss = -np.sum(y_onehot * np.log(probs + 1e-15)) / n_samples
            # X(n_sample,d_feature) prob(n,t_class) y(n,t)
            # 对于每个特征，计算其对所有类别的误差的贡献。
            # 每个样本的误差与其对应特征的值相乘，最终累加得到每个特征的梯度
            # divide by n，希望梯度不依赖于样本的数量,梯度平均。
            dW = np.dot(X.T, (probs - y_onehot)) / n_samples  # weight gradient
            db = np.sum(probs - y_onehot, axis=0, keepdims=True) / n_samples  # bias gradient

            # update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            if self.verbose and i % 50 == 0:
                print(f"Iteration {i}, loss: {loss:.4f}")

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


# ------------------- test -------------------
if __name__ == '__main__':
    # 1. load datasets
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. ->0,1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. train
    model = MyLogisticRegression(learning_rate=0.1, num_iter=300, verbose=True)
    model.fit(X_train, y_train)

    # 5. predict
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"accuracy of test datasets: {accuracy:.4f}")
