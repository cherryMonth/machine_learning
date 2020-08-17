def recall(y_true, y_pred):
    """
    针对原始数据集：给定的测试数据集，样本中多少正样本被正确预测了
    :param y_true: 预测数据集
    :param y_pred: 测试数据集
    :return:
    """
    tp_fn = 0
    tp = 0
    for index in range(len(y_true)):
        # 如果预测正确，那么正确的总数+1
        if y_true[index] == 1:
            tp_fn += 1
            if y_pred[index] == 1:
                tp += 1
    return tp / tp_fn


if __name__ == '__main__':
    pred = [1, -1, -1, -1, 1, 1, 1, -1, 1, 1]
    test = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1]
    print("precision is {}".format(recall(test, pred)))
