import numpy as np


class Smote:
    def __init__(self, k, n):
        """
        :param k: 设置每个少数样本中近邻的个数
        :param n: 设置从近邻样本中采样的个数
        """
        self.n = n
        self.k = k

    def fit_transform(self, dataset, label, target_label):
        """
        从数据集中对少数样本进行过采样
        :param dataset: 原始特征矩阵
        :param label: 特征对应的标签
        :param target_label: 需要进行过采样的标签
        :return:
        """
        # 从样本数据集中找到少数样本的标签索引
        index = np.where(label == target_label)[0]

        # 找到这些少数样本的特征矩阵
        rate_sample = np.array(dataset[index])

        # 计算这些少数样本之间的欧式距离, 如果少数样本的行数为n，则距离矩阵的形状为n * n
        # 对于其中的任意元素 d_ij，代表第i个样本到第j个样本的欧式距离
        distance_matrix = np.sqrt(
            -2 * rate_sample.dot(rate_sample.T) + np.sum(np.square(rate_sample), axis=1, keepdims=True)
            + np.sum(np.square(rate_sample), axis=1))

        # 由于对角线元素到自身的欧式距离为0，所以要替换成无穷大，用以过滤掉这些元素
        diagonal = np.arange(distance_matrix.shape[0])
        distance_matrix[diagonal, diagonal] = np.Inf

        # 对每一行的距离向量进行排序，得到该向量的距离排行
        index_sort = np.argsort(distance_matrix)

        # 如果top k向量的长度小于用户指定k的个数直接返回距离向量，否则只取top k个元素
        get_top_k_index = index_sort if index_sort.shape[1] < self.k else index_sort[:, :self.k]

        # 从top k个元素中进行随机采样得到n个样本
        get_top_n_index_sample = np.array(
            list(map(lambda _: np.random.choice(_, self.n, replace=False), get_top_k_index)))
        result = np.array([])

        # 对每个少数样本进行迭代合成新样本
        for n_index in range(get_top_n_index_sample.shape[0]):
            # 得到距离目标少数样本的n个top k样本
            sample = rate_sample[get_top_n_index_sample[n_index]]

            # 得到目标少数样本
            target_sample = rate_sample[n_index]

            # 合成新样本
            new_vec = target_sample + np.random.random() * (sample - target_sample)
            if result.shape[0] == 0:
                result = new_vec
            else:
                # 把新样本加入到结果列表中
                result = np.concatenate((result, new_vec))
        return result


if __name__ == '__main__':
    # 生成二维高斯分布
    x = np.random.normal(loc=0, scale=1, size=(10000, 2))
    y = np.ones(10000)
    x_abs = np.abs(x)
    # 如果两个维度都超过了两个标准差，则把样本记为异常样本
    neg_index = np.where((x_abs[:, 0] >= 2) & (x_abs[:, 1] >= 2))  # 设置超过两个标准差的异常点
    print(x[neg_index])
    y[neg_index] = 0  # 把异常样本的标签设置为0
    # 对数据进行过采样
    print(Smote(10, 5).fit_transform(x, y, 0))
