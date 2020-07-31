import numpy as np
import importlib
import sys

"""
参考链接
http://sofasofa.io/forum_main_post.php?postid=1003214
https://blog.csdn.net/randompeople/article/details/95042487
"""

sys.path.append('../model/')
LinearRegression = importlib.import_module('multiply_linear_regression').LinearRegression


class Boosting:

    def __init__(self, estimator, n_iterates=10, alpha=0.8):
        self.estimator = estimator
        self.n_iterates = n_iterates
        self.model_list = []
        self.model_weight = []
        self.data_weight = np.array([])
        self.alpha = alpha

    def fit(self, data, label):
        """
        模型拟合过程，实现原理参考对应的链接
        :param data: 特征矩阵
        :param label: 标签
        :return:
        """
        # 设置初始的数据分布的采样权重，此时都相等
        self.data_weight = np.ones((data.shape[0], 1)) / data.shape[0]

        # 记录数据集的索引
        index = np.arange(0, data.shape[0], 1)
        # 进行迭代求解
        for i in range(self.n_iterates):
            # 根据数据权重进行采样, 注意bagging是有放回，boosting是无放回
            # https://zhuanlan.zhihu.com/p/47922595
            sub_samping = np.random.choice(index, int(index.size * self.alpha), replace=False,
                                           p=self.data_weight.reshape(-1, ).tolist())

            train_x = data[sub_samping]
            train_y = label[sub_samping]
            lr = LinearRegression(method='sgd')  # 进行弱学习模型训练
            lr.fit(train_x, train_y)

            self.model_list.append(lr)  # 存储该弱学习模型

            pred = lr.predict(train_x)
            loss = np.abs(train_y - pred)  # 计算每个样本的相对误差 (线性误差)

            # 计算模型在训练集上的回归误差率
            et_i = loss / np.max(loss)
            et = np.dot(et_i, self.data_weight[sub_samping])

            # 把模型的权重加入到列表中
            at = et / (1 - et)
            self.model_weight.append(at)

            # 更新样本的权重
            self.data_weight[sub_samping] = self.data_weight[sub_samping] * np.power(at, 1 - et_i).reshape(-1, 1)

            # 权重归一化
            self.data_weight = self.data_weight / self.data_weight.sum()

    def predict(self, data):
        # 返回权重为中位数的模型（过低的模型容器欠拟合，过高的模型容易过拟合）
        median = np.median(self.model_weight)
        # 由于结果都是浮点数，所以要使用差值小于一个小量代表两者相同
        median_index = np.where(np.array(self.model_weight) - median <= 1e-3)[0]
        result = []
        for index in median_index:
            # 每个模型的预测结果需要乘以对应的权重
            result.append(np.log(1 / self.model_weight[index]) * self.model_list[index].predict(data))
        # 把每个模型预测的结果相加为最终的强模型
        return np.sum(result, axis=0)


if __name__ == '__main__':
    x_test = np.random.random((1000, 2))
    weight = np.array([1, 10])
    y_test = np.sum(weight * x_test, axis=1)
    boosting = Boosting(LinearRegression(method='sgd'))
    boosting.fit(x_test, y_test)
    from sklearn.metrics import mean_squared_error

    y_pred = boosting.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
