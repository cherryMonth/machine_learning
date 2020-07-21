"""
公式：
theta=(X.T*X)-1*X.T*y

数据的准备:
theta0 theta1
x0  工资    额度
1   4000    20000
1   8000    50000
1   5000    30000
1   10000   70000
1   12000   60000
1   15000    ?
"""
import numpy as np


x = np.array([[1, 4],
              [1, 8],
              [1, 5],
              [1, 10],
              [1, 12]])
y = np.array([20, 50, 30, 70, 60])


class LinearRegression(object):

    def __init__(self, method, alpha=0.001, n_iterates=1000):
        self.theta = None
        self.method = method
        self.alpha = alpha
        self.n_iterates = n_iterates

    # 训练模型
    def fit(self, x, y):
        # 求解theta  这个地方的theta是一个向量
        if self.method == 'sgd':
            if not self.theta:
                self.theta = np.random.random((x.shape[1]))
            m = x.shape[0]
            for _ in range(self.n_iterates):
                gradient = 1 / m * x.T.dot(x.dot(self.theta) - y)
                self.theta = self.theta - self.alpha * gradient
        elif self.method == 'liner':
            self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        else:
            raise Exception("no such method!")

    # 预测
    def predict(self, x):
        return x.dot(self.theta)

    def __repr__(self):
        return str(self.theta)


if __name__ == '__main__':
    x_test = np.random.random((1000, 2))
    weight = np.array([1, 10])
    y_test = np.sum(weight * x_test, axis=1)
    model = LinearRegression(method='sgd')
    model.fit(x_test, y_test)
    print(model.theta)
    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
