import numpy as np

"""
https://blog.csdn.net/qq_22238533/article/details/78666436
AUC的定义是ROC曲线的面积，由于该问题与预测得到正样本的概率大于负样本概率的概率等价，所以用后者代替面积
"""


def get_average_rank(value, iteration):
    """
    计算阈值相等的索引之和的平均值
    :param value:
    :param iteration:
    :return:
    """
    count = 0
    index_count = 0
    for index in range(1, len(iteration) + 1):
        if value == iteration[index - 1]:
            index_count += index
            count += 1
    return index_count / count


def auc(y_true, scores):
    """
    计算正样本大于负样本的数量，该问题可以简化成以下步骤：
    1、先对预测的置信度进行排序得到索引
    2、把正样本的索引之和相加减去重复计数的正样本数量就是样本对中正样本大于负样本的数量
    :param y_true:
    :param scores:
    :return:
    """
    arg_index = np.argsort(scores)
    rank_sum = 0
    tp = 0
    tn = 0
    for index in range(len(arg_index)):
        if y_true[arg_index[index]] == 1:
            rank_sum += get_average_rank(scores[arg_index[index]], scores[arg_index])
            tp += 1
        else:
            tn += 1
    return (rank_sum - tp * (tp + 1) / 2) / (tp * tn)


if __name__ == '__main__':
    test_true = np.array([1, 1, 0, 0, 1, 1, 0])
    test_scores = np.array([0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.3])
    print(auc(test_true, test_scores))
    test_true = np.array([0, 0, 1, 1])
    test_scores = np.array([0.1, 0.4, 0.35, 0.8])
    print(auc(test_true, test_scores))
