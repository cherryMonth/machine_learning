"""
根据最小二乘法推导求得的公式：
w=sigma(xi*yi-xi*y_mean)/sigma(xi*xi-xi*x_mean)
w=sigma((xi-x_mean)*(yi-y_mean))/sigma(xi-x_mean)**2
b=y_mean-w*x_mean
"""
import numpy as np

"""
数据的准备:
工资    额度
4000    20000
8000    50000
5000    30000
10000   70000
12000   60000
15000    ?
"""
x = np.array([[4],
              [8],
              [5],
              [10],
              [12]])
y = np.array([20, 50, 30, 70, 60])


class LinearRegression(object):

    # 训练模型
    def fit(self, x, y):
        # 样本个数
        m = np.size(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        # 分子的变量
        numerator = 0
        # 分母的变量
        dinominator = 0
        for i in range(m):
            numerator += x[i] * y[i] - x[i] * y_mean
            dinominator += x[i] ** 2 - x[i] * x_mean
        self.w = numerator / dinominator
        self.b = y_mean - self.w * x_mean

    # 预测
    def predict(self, x):
        return self.w * x + self.b
    
    def __repr__(self):
        return "weight: {}, bias : {}".format(self.w, self.b)


if __name__ == "__main__":
    model = LinearRegression()
    model.fit(x, y)
    print("w:", model.w, "b:", model.b)
    print(model.predict(15))
