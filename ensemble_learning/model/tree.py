import numpy as np


def calc_shannon_ent(label):
    """
    计算数据集中的信息熵
    :param label: 原始的标签
    :return:
    """

    # 计算每个种类的占比
    prob = np.unique(label, return_counts=True)[1] / label.size
    # 计算信息熵
    return - prob.dot(np.log2(prob))


def splitDataSet(data_set, label, colume, value):
    """
    给定数据集和列以及属性，找到满足这些条件的数据集
    :param data_set:
    :param label:
    :param colume:
    :param value:
    :return:
    """
    index = np.where(data_set[:, colume] == value)
    feature = np.delete(data_set[index], colume, axis=1)
    return feature, label[index]


def calc_v_shannon_ent(data_set, label, column, value):
    """
    计算数据集按照某列的某个属性划分时得到的信息熵
    :param label:
    :param data_set:
    :param column:
    :param value:
    :return:
    """
    subDataSet = splitDataSet(data_set, label, column, value)
    prob = subDataSet[0].size / data_set.size
    return column, prob * calc_shannon_ent(subDataSet[1])


def chooseBestFeatureToSplit(data_set, label):
    """
    从数据集中找到分割代价最小的特征并返回下标
    :param data_set: 特征矩阵
    :param label: 标签
    :return:
    """
    baseEntropy = calc_shannon_ent(label)
    bestInfoGain = 0.0
    bestFeature = -1
    index = 0

    for featList in data_set.T:
        # 对每列数据进行迭代，计算出每个属性划分的代价
        uniqueVals = np.unique(featList)
        newEntropy = np.sum(list(map(lambda value: calc_v_shannon_ent(data_set, label, index, value), uniqueVals)))
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = index
        index += 1
    return bestFeature


def createTree(data_set, labels, feature):
    """
    创建一颗分类树
    :param data_set:
    :param labels:
    :param feature:
    :return:
    """
    # 如果样本的所有类别相同，这返回这个类别
    if np.max(np.unique(labels, return_counts=True)[1]) == labels.size:
        return labels[0]

    # 找到划分代价最小的特征
    bestFeat = chooseBestFeatureToSplit(data_set, labels)
    bestFeatLabel = feature[bestFeat]
    tree = {bestFeatLabel: {}}
    # 从已经特征名单中去除已经划分的特征
    feature = np.delete(feature, bestFeat)
    uniqueVals = np.unique(data_set[:, bestFeat])
    for index in range(uniqueVals.size):
        # 对每一个属性进行递归求子树
        tree[bestFeatLabel][uniqueVals[index]] = createTree(
            *splitDataSet(data_set, labels, bestFeat, uniqueVals[index]),
            feature)
    return tree


def predict(model, x_pred, feature):
    firstStr = list(model.keys())[0]  # 找到当前根节点的标签
    secondDict = model[firstStr]  # 得到其子树
    index = np.where(feature == firstStr)  # 找到在该特征下的属性
    key = x_pred[index][0]  # 找到测试数据该特征下的属性值
    valueOfFeat = secondDict[key]  # 从属性值找到符合要求的子树
    if isinstance(valueOfFeat, dict):  # 如果结果是字典代表要在子树的基础上继续查找
        classLabel = predict(valueOfFeat, x_pred, feature)
    else:  # 查询成功返回结果
        classLabel = valueOfFeat
    return classLabel


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y = np.array([1, 1, 0, 0, 0])
    feature_list = np.array(['no surfacing', 'flippers'])
    dt = createTree(x, y, feature_list)
    x_test = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y_pred = list(map(lambda pred: predict(dt, pred, feature_list), x_test))
    print(y_pred)
