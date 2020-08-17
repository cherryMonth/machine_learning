import numpy as np


def mae(y_true, y_pred):
    """
    平均绝对误差
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.sum(abs(y_true - y_pred)) / y_true.size


if __name__ == '__main__':
    test = np.array([1, 4, 5, 4, 5])
    pred = np.array([1, 1, 2, 3, 5])
    print("mae = {}".format(mae(test, pred)))
