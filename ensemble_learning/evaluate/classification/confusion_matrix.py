def confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return:
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(len(y_true)):
        if y_true[index] == 1:
            if y_pred[index] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[index] == -1:
                tn += 1
            else:
                fp += 1
    return tp, fp, tn, fn


if __name__ == '__main__':
    pred = [1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1]
    test = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1]
    print(confusion_matrix(test, pred))
