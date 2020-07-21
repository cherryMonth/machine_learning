import numpy as np
from copy import deepcopy
import importlib
import sys

sys.path.append('../model/')

LinearRegression = importlib.import_module('multiply_linear_regression').LinearRegression
SimpleLR = importlib.import_module('simple_linear_regression').LinearRegression


class Bagging:

    def __init__(self, n_estimators, estimator, rate=1.0, random=0):
        """
        :param n_estimators: 分类器个数
        :param estimator: 分类器
        :param rate: 样本分割比例
        :param random: 随机种子
        """
        self.n_estimators = n_estimators
        # 返回模型的深拷贝
        self.estimators = list(map(lambda lam: deepcopy(estimator), range(n_estimators)))
        self.rate = rate
        self.random = random

    def sub_sampling(self, data, label):
        """
        对样本进行上采样
        :param data: 特征字段
        :param label: 标签字段
        :return: 返回${n_estimators}组包含${len(data) * self.rate}个训练样本的训练集
        """
        result = list()
        index = np.arange(0, data.shape[0], 1)
        for n in range(self.n_estimators):
            # 有放回采样${len(data) * self.rate}个样本
            sub_index = np.random.choice(index, int(index.size * self.rate), replace=True)
            sub_x = data[sub_index]
            sub_y = label[sub_index]
            result.append((sub_x, sub_y))

        return result

    def fit(self, data, label):
        """
        模型拟合
        :param data:
        :param label:
        :return:
        """
        train_set = self.sub_sampling(data, label)
        for index in range(self.n_estimators):
            self.estimators[index].fit(train_set[index][0], train_set[index][1])

    def predict(self, x_pred):

        result = np.array([])

        for feature in x_pred:  # 对每一行特征进行迭代
            vote_array = np.array([])
            # 记录模型的预测结果进行投票
            for estimator in self.estimators:
                y_pred = estimator.predict(feature)
                vote_array = np.append(vote_array, y_pred)

            # 取预测各个模型预测的平均值的作为模型的预测结果
            result = np.append(result, np.mean(vote_array))

        return result


if __name__ == '__main__':
    x_test = np.random.random((1000, 2))
    weight = np.array([1, 10])
    y_test = np.sum(weight * x_test, axis=1)
    bagging = Bagging(10, LinearRegression(method='sgd'), 0.8)
    bagging.fit(x_test, y_test)
    from sklearn.metrics import mean_squared_error

    y_pred = bagging.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
