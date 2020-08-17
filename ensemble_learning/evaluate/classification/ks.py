from roc import roc
import numpy as np


def ks(y_true, y_pred):
    """
    可以使用roc结果的fpr和tpr简化ks的计算
    :param y_true:
    :param y_pred:
    :return:
    """
    fpr, tpr, thresholds = roc(y_true, y_pred)
    # 累计good%-累计bad%，这些绝对值取最大值即得此评分卡的KS值。
    return max(abs(np.array(tpr) - np.array(fpr)))


if __name__ == '__main__':
    test = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pred = [0.5, 0.6, 0.7, 0.6, 0.6, 0.8, 0.4, 0.2, 0.1, 0.4, 0.3, 0.9]
    print("ks is {}".format(ks(test, pred)))
