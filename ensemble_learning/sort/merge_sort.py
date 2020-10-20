def sort(num_list, begin, length):
    mid = (begin + length) // 2

    i = begin
    j = mid
    tmp = []

    while i < mid and j < length:
        if num_list[i] < num_list[j]:
            tmp.append(num_list[i])
            i += 1
        else:
            tmp.append(num_list[j])
            j += 1

    while i < mid:
        tmp.append(num_list[i])
        i += 1

    while j < length:
        tmp.append(num_list[j])
        j += 1

    j = 0
    for i in range(begin, length):
        num_list[i] = tmp[j]
        j += 1


def merge(num_list, begin, length):
    if begin < length - 1:
        mid = (begin + length) // 2
        merge(num_list, begin, mid)
        merge(num_list, mid, length)
        sort(num_list, begin, length)


test = [5, 4, 3, 2, 1, 2, 2, 2, 2, 2, -2, 4, 5, 4, 5]
merge(test, 0, len(test))
print(test)
