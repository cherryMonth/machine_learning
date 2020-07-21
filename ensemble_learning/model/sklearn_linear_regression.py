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
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[2, 4],
              [2, 8],
              [2, 5],
              [2, 10],
              [2, 12]])
y = np.array([20, 50, 30, 70, 60])

# 构建模型
model = LinearRegression(fit_intercept=False)
# 训练
model.fit(x, y)
"""
w: [5.71428571] b: [1.42857143]
[87.14285714]
"""
print(model.coef_, model.intercept_)
# 预测
y_pred = model.predict([[1, 15]])
print(y_pred)
