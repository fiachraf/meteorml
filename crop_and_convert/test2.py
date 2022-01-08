list = [1,2,3,4,5,6,7,8,9]

# for element in list:
#     if element == 5:
#         element = 1
#     print(element)
# print(list)

#output:
# 1, 2, 3, 4, 1, 6, 7, 8, 9
#[1, 2, 3, 4, 5, 6, 7, 8, 9]


# for index, element in enumerate(list):
#     if element == 5:
#         list[index] = 1
#     print(element)
# print(list)

#output:
# 1, 2, 3, 4, 5, 6, 7, 8, 9
#[1, 2, 3, 4, 1, 6, 7, 8, 9]


def test_rec(num_loops):
    test_list = []
    if num_loops != 0:
        test_list.append(num_loops)
        return test_rec(num_loops - 1)
    print(f"num_loops: {num_loops}")
    print(f"test_list: {test_list}")
    return

print(f"test_rec(10): {test_rec(10)}")
