import numpy as np
from tree import createTree, generate_data, predict

"""
参考链接
http://sofasofa.io/forum_main_post.php?postid=1003214
https://blog.csdn.net/randompeople/article/details/95042487
"""


class Boosting:

    def __init__(self, feature_list, n_iterates=10, alpha=0.8):
        self.n_iterates = n_iterates
        self.model_list = []
        self.model_weight = []
        self.data_weight = np.array([])
        self.alpha = alpha
        self.feature_list = feature_list

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
            sub_samping = np.random.choice(index, int(self.data_weight.shape[0] * self.alpha), replace=False,
                                           p=self.data_weight.reshape(-1, ).tolist())
            train_x = data[sub_samping]
            train_y = label[sub_samping]
            dt = createTree(train_x, train_y, self.feature_list)  # 进行弱学习模型训练

            self.model_list.append(dt)  # 存储该弱学习模型

            pred = list(map(lambda _: predict(dt, _, self.feature_list), train_x))
            # 计算模型在训练集上的误差率 (即预测错误的样本权重相加，相同为0，不同为1)
            pred_error = np.ones((len(pred), 1))
            pred_error[pred == train_y] = 0
            et = pred_error.T.dot(self.data_weight[sub_samping])

            # 把模型的权重加入到列表中
            at = 0.5 * np.log((1 - et) / et)
            self.model_weight.append(at)

            # 更新样本的权重
            self.data_weight[sub_samping] = self.data_weight[sub_samping] * np.exp(- at * train_y * pred).reshape(-1, 1)

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
            pred = list(map(lambda _: predict(self.model_list[index], _, self.feature_list), data))
            result.append(self.model_weight[index] * pred)
        # 把每个模型预测的结果相加为最终的强模型
        return np.sign(np.sum(result, axis=0)).astype(int).reshape(-1)


if __name__ == '__main__':
    feature_list = np.array(['first', 'second', 'third', 'fourth', 'fifth'])
    x, y = generate_data()
    boosting = Boosting(feature_list)
    boosting.fit(x, y)

    x, y = generate_data()
    y_pred = boosting.predict(x)
    from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print(roc_auc_score(y, y_pred))
