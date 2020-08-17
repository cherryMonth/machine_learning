import numpy as np


def mse(y_true, y_pred):
    """
    均方误差
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.sum(np.power(y_true - y_pred, 2)) / y_true.size


if __name__ == '__main__':
    test = np.array([1, 4, 5, 4, 5])
    pred = np.array([1, 1, 2, 3, 5])
    print("mae = {}".format(mse(test, pred)))
