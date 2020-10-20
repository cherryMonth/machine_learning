def select_sort(num_list):
    for i in range(len(num_list) - 1):
        k = i
        # 从当前索引的后续中找到最小的元素
        for j in range(i + 1, len(num_list)):
            if num_list[j] < num_list[k]:
                k = j
        if k != i:
            tmp = num_list[i]
            num_list[i] = num_list[k]
            num_list[k] = tmp


test = [5, 4, 3, 2, 1, 2, 2, 2, 2, 2, -2, 4, 5, 4, 5]
select_sort(test)
print(test)
