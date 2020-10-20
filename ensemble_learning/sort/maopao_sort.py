def maopao(num_list):
    for i in range(len(num_list) - 1):
        for j in range(len(num_list) - 1):
            if num_list[j + 1] < num_list[j]:
                tmp = num_list[j + 1]
                num_list[j + 1] = num_list[j]
                num_list[j] = tmp


test = [5, 4, 3, 2, 1, 2, 2, 2, 2, 2, -2, 4, 5, 4, 5]
maopao(test)
print(test)
