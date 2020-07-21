import numpy as np
import importlib
import sys
import copy

sys.path.append('../model/')
LinearRegression = importlib.import_module('multiply_linear_regression').LinearRegression

"""
stacking集成的原理类似于多层感知机
"""


class Stacking:

    def __init__(self, n_estimators, meta_model, model, split=0.8, random=0):
        """
        :param n_estimators: 元模型的数量
        :param random: 随机数种子
        :param split: 训练集和测试集分割比例，训练集用于元模型进行训练，测试集用于元模型生成给决策模型的数据
        """
        self.n_estimators = n_estimators
        self.random = random
        self.split = split
        self.meta_model = list(map(lambda x: copy.deepcopy(meta_model), range(n_estimators)))
        self.model = model

    def predict(self, x_pred):
        """
        把元模型的输出作为最终模型的特征
        :param x_pred: 原始数据
        :return:
        """
        dataset_blend_feature = np.zeros((x_pred.shape[0], self.n_estimators))
        for index, estimator in enumerate(self.meta_model):
            dataset_blend_feature[:, index] = estimator.predict(x_pred)

        return self.model.predict(dataset_blend_feature)

    def fit(self, data, label):
        """

        :param data: 特征矩阵
        :param label: 标签矩阵
        :return:
        """
        index = np.arange(0, data.shape[0], 1)

        # 数据集进行打乱
        np.random.shuffle(index)

        # 划分训练集和测试集
        split_index = int(index.size * self.split)

        train_set_index = index[0: split_index]
        test_set_index = index[split_index:]

        # 创建零矩阵用于决策模型训练，内容包含， n个模型对于m行样本的预测值，所以形状为 m * n。
        dataset_blend_train = np.zeros((test_set_index.size, self.n_estimators))

        for model_index, model in enumerate(self.meta_model):
            model.fit(data[train_set_index], label[train_set_index])
            dataset_blend_train[:, model_index] = model.predict(data[test_set_index])

        self.model.fit(dataset_blend_train, label[test_set_index])


if __name__ == '__main__':
    x_test = np.random.random((1000, 2))
    weight = np.array([1, 10])
    y_test = np.sum(weight * x_test, axis=1)
    stack = Stacking(5, LinearRegression(method='sgd'), LinearRegression(method='sgd'))
    stack.fit(x_test, y_test)

    from sklearn.metrics import mean_squared_error
    y_pred = stack.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
