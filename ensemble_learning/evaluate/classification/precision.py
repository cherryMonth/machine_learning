def precision(y_true, y_pred):
    """
    针对预测结果：计算给定的测试数据集，预测结果为真占全部标签结果为真的比例
    :param y_true: 预测数据集
    :param y_pred: 测试数据集
    :return:
    """
    tp_fp = 0
    tp = 0
    for index in range(len(y_true)):
        # 如果预测正确，那么正确的总数+1
        if y_pred[index] == 1:
            tp_fp += 1
            if y_true[index] == 1:
                tp += 1
    return tp / tp_fp


if __name__ == '__main__':
    pred = [1, -1, 1, -1, 1, 1, 1, -1, 1, 1]
    test = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1]
    print("precision is {}".format(precision(test, pred)))
