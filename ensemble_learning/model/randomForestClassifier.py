import numpy as np
from tree import createTree, generate_data, predict


class RandomForestClassifier:
    def __init__(self, n_estimators, max_features, sample_rate, feature_list):
        """
        :param n_estimators: 决策树个数
        :param max_features: RF划分时考虑的最大特征数，如果是auto：则意味着划分时最多考虑√N个特征，如果是log2意味着最多考虑log2个特征
        :param sample_rate: 采样比例
        :param feature_list: 特征的列名
        """
        self.n_estimators = n_estimators  # 决策树个数
        self.max_features = max_features  # 设置子决策树列树的个数的方式
        self.tree_list = []  # 保存决策树的模型列表
        self.tree_feature = []  # 每个树对应的训练数据集的列索引
        self.sample_rate = sample_rate  # 训练时采样的比例
        self.feature_size = None  # 根据特征的采样方式得到特征的数量
        self.feature_list = feature_list  # 记录各个决策树所对应样本的列索引

    def sub_sampling(self, data, label):
        """
        采样bootstrap采样
        :param data: 特征矩阵
        :param label: 特征标签
        :return:
        """
        index = np.arange(0, data.shape[0], 1)
        # 有放回采样${len(data) * self.sample_rate}个样本
        sub_index = np.random.choice(index, int(index.size * self.sample_rate), replace=True)
        sub_x = data[sub_index]
        sub_y = label[sub_index]
        return sub_x, sub_y

    def fit(self, data, label):
        # 得到决策树的列个数
        if self.max_features == 'auto':
            self.feature_size = np.rint(np.sqrt(data.shape[1])).astype(int) + 1
        else:
            self.feature_size = np.rint(np.log2(data.shape[1])).astype(int) + 1

        # 记录特征矩阵所有特征的索引
        feature_index = np.arange(0, data.shape[1], 1)

        for index in range(self.n_estimators):
            sub_x, sub_y = self.sub_sampling(data, label)
            # 从特征索引列表中随机抽取特征作为决策树训练的数据集
            feature_sample_index = np.random.choice(feature_index, self.feature_size, replace=False)
            # 保存决策树的列索引
            self.tree_feature.append(feature_sample_index)
            self.tree_list.append(
                createTree(sub_x[:, feature_sample_index], sub_y, self.feature_list[feature_sample_index]))

    def predict(self, x_pred):
        result = np.array([])

        for feature_vec in x_pred:  # 对每一行特征进行迭代
            vote_array = np.array([])
            # 记录模型的预测结果进行投票
            for index in range(self.n_estimators):
                pred = predict(self.tree_list[index], feature_vec[self.tree_feature[index]],
                               self.feature_list[self.tree_feature[index]])
                vote_array = np.append(vote_array, pred)
            # 取预测各个模型预测的投票结果的作为模型的预测结果
            label_class, counts = np.unique(vote_array.astype(int), return_counts=True)
            most_label_index = np.argmax(counts)
            result = np.append(result, label_class[most_label_index])

        return result


if __name__ == '__main__':
    feature_list = np.array(['first', 'second', 'third', 'fourth', 'fifth'])
    rf = RandomForestClassifier(9, 'auto', 0.8, feature_list)
    x, y = generate_data()
    rf.fit(x, y)
    from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

    x, y = generate_data()
    y_pred = rf.predict(x)
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred))
    print(roc_auc_score(y, y_pred))
