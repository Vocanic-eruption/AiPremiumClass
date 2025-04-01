from sklearn.datasets import make_classification;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
import numpy as np;


def forword(bias, x_train, theta):
    z = np.dot(theta, x_train.T) + bias;
    # sigmoid 求估值
    y_hat = 1 / (1 + np.exp(-z));
    return y_hat;

def loss_funtion(y_hat, y_train):
    e = 1e-8;
    loss = -y_train * np.log(y_hat) - (1 - y_train) * np.log(1 - y_hat)
    return loss;

def gradient(y_train, y_hat, x_train):
    m = x_train.shape[-1];
    delta_theta = np.dot(y_hat - y_train, x_train) / m;
    delta_bias = np.mean(y_hat - y_train);
    return delta_theta, delta_bias;

def logicReturn(*, bias = 0., epochs = 3000):
    # 测试数据过拟合
    # X,y = load_iris(return_X_y=True);
    # x_train = X[:100];
    # y_train = y[:100];
    X, y = make_classification(n_samples=300,n_features=15)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    l_rate = 1e-3;
    theta = np.random.randn(1, 15);
    for i in range(epochs):
        y_hat = forword(bias, X_train, theta);
        loss = loss_funtion(y_hat, y_train);
        loss_mean = np.mean(loss);
        if i % 100 == 0:
            acc = np.mean(np.round(y_hat) == y_train);
            print(f"step:{i},loss:{loss_mean},acc:{acc}");
        delta_theta, delta_bias = gradient(y_train, y_hat, X_train);
        theta = theta - l_rate * delta_theta;
        bias = bias - l_rate * delta_bias;
    for i in range(1,20):
        test_model(X_test, y_test, bias, X_train, theta)

def test_model(X_test, y_test, bias, theta):
    idx = np.random.randint(len(X_test.shape))
    x = X_test[idx]
    y = y_test[idx]
    predict = np.round(forword(bias, x, theta))
    print(f"truely:{y};predict:{predict}\n")

