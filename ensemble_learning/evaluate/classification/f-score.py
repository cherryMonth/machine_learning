from precision import precision
from recall import recall


def f1_score(y_true, y_pred):
    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 2 * pr * re / (pr + re)


if __name__ == '__main__':
    pred = [1, -1, 1, -1, 1, 1, 1, -1, 1, 1]
    test = [-1, -1, 1, -1, 1, 1, 1, -1, -1, 1]
    print("f1 is {}".format(f1_score(pred, test)))
