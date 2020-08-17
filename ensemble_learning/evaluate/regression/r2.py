import numpy as np
from mse import mse


def r2(y_true, y_pred):
    """
    r2 = 1 - 模型均方误差 / 样本点平均值误差
    当预测拟合的结果不如盲猜的平均值好时，结果为负
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1 - mse(y_true, y_pred) / np.var(y_pred)


if __name__ == '__main__':
    test = np.array([1, 1.5, 2, 4, 5])
    pred = np.array([1, 1, 2, 3, 5])
    print("r2 = {}".format(r2(test, pred)))
