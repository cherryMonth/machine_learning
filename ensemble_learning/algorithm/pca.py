import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.feature_vec = None
        self.feature = None

    def fit(self, dataset):
        """
        计算得到数据集的协方差矩阵的特征top k特征向量
        :param dataset:
        :return:
        """
        # 数据集减去列向量的平均值
        data_adjust = dataset - np.mean(dataset, axis=0)
        # 计算协方差矩阵
        data_cov = np.cov(data_adjust.T)
        # 得到特征值和特征向量
        feature, feature_vec = np.linalg.eig(data_cov)
        index = np.argsort(feature)  # 特征值按top k 排序得到特征向量
        self.feature_vec = feature_vec[index[:self.n_components]]
        self.feature = feature[index[:self.n_components]]
        return self

    def transform(self, dataset):
        """
        数据集减去均值与特征矩阵的乘积即为投影后的数据
        :param dataset:
        :return:
        """
        data_adjust = dataset - np.mean(dataset, axis=0)
        return data_adjust.dot(self.feature_vec.T)


if __name__ == '__main__':
    data = np.random.random((10, 4))
    a = PCA(3).fit(data)
    print(a.transform(data))
