def head_adjust(num_list, index, heap_size):
    """
    构建大顶堆
    :param num_list:
    :param index:
    :param heap_size:
    :return:
    """
    _max = index
    left = 2 * index
    right = 2 * index + 1
    if left < heap_size and num_list[left] > num_list[_max]:
        _max = left
    if right < heap_size and num_list[right] > num_list[_max]:
        _max = right

    if _max != index:
        tmp = num_list[index]
        num_list[index] = num_list[_max]
        num_list[_max] = tmp
        head_adjust(num_list, _max, heap_size)


def head_sort(num_list):
    length = len(num_list)
    i = length // 2 - 1
    while i > 0:
        head_adjust(num_list, i, length)
        i -= 1

    i = length - 1
    while i > 0:
        # 第一次先找到最大的元素，然后让他与首元素交换
        tmp = num_list[i]
        num_list[i] = num_list[0]
        num_list[0] = tmp

        # 交换完成之后，缩小构建范围，继续构建大顶堆
        head_adjust(num_list, 0, i)
        i -= 1


test = [5, 4, 3, 2, 1, 2, 2, 2, 2, 2, -2, 4, 5, 4, 5]
head_sort(test)
print(test)
