from mse import mse
import numpy as np


def rmse(y_true, y_pred):
    """
    均方根误差
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.sqrt(mse(y_true, y_pred))


if __name__ == '__main__':
    test = np.array([1, 4, 5, 4, 5])
    pred = np.array([1, 1, 2, 3, 5])
    print("mae = {}".format(rmse(test, pred)))
