def quick(num_list):
    # 先写终止条件
    if len(num_list) < 2:
        return num_list
    index = num_list[0]
    # 找到所有小于枢轴的列表
    left = [num for num in num_list[1:] if num < index]
    # 找到所有大于枢轴的列表
    right = [num for num in num_list[1:] if num >= index]
    return quick(left) + [index] + quick(right)


test = [5, 4, 3, 2, 1, 2, 2, 2, 2, 2, -2, 4, 5, 4, 5]

print(quick(test))
