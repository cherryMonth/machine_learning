def roc(y_true, scores):
    """
    y_pred
    :param y_true: 数据集真正的标签列
    :param scores: 模型预测结果的置信度，如果置信度大于给定的阈值则为1，否则为0
    :return:
    """
    fpr_list = []
    tpr_list = []
    thresholds = []
    # 把置信度从大到小进行排序，并依次把排序后的元素作为阈值
    for threshold in sorted(scores, reverse=True):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for index in range(len(y_true)):
            if y_true[index] == 1:
                # 当预测结果大于等于阈值且样本真实标签为1，则正阳+1
                if scores[index] >= threshold:
                    tp += 1
                # 当预测结果小于阈值且样本真实标签为1，则假阴+1
                else:
                    fn += 1
            else:
                # 当预测结果大于等于阈值且样本真实标签为-1，则假阳+1
                if scores[index] >= threshold:
                    fp += 1
                # 当预测结果小于阈值且样本真实标签为-1，则真阴+1
                else:
                    tn += 1
        fpr = fp / (fp + tn)  # 计算此时的假阳率
        tpr = tp / (tp + fn)  # 计算此时的真阳率
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        thresholds.append(threshold)  # 记录此时的阈值
    return fpr_list, tpr_list, thresholds


if __name__ == '__main__':
    test = [-1, -1, 1, 1]
    test_scores = [0.1, 0.4, 0.35, 0.8]
    print(roc(test, test_scores))
