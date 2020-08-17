def accuracy(y_test, y_pred):
    """
    计算给定的测试数据集，预测正确的分类与总样本数值比
    :param y_pred: 预测数据集
    :param y_test: 测试数据集
    :return:
    """
    length = len(y_pred)
    correct_num = 0
    for index in range(length):
        # 如果预测正确，那么正确的总数+1
        if y_pred[index] == y_test[index]:
            correct_num += 1
    return correct_num / length


if __name__ == '__main__':
    pred = [1, -1, 1, -1, 1, 1, 1, -1, 1, 1]
    test = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1]
    print("accuracy is {}".format(accuracy(test, pred)))
