def transpose_matrix(matrix):
    convert = list(zip(*matrix))
    transpose_matrix_ = list(map(list, convert))
    return transpose_matrix_

def multiplication_matrix(matrix_1, matrix_2):
    transpose_matrix_2 = transpose_matrix(matrix_2)
    list_res = []
    for ind in range(len(matrix_1)):
        sub_list = []
        res_to_matrix = 0
        for ind_2 in range(len(matrix_2[0])):
            for el_1, el_2 in zip(matrix_1[ind], transpose_matrix_2[ind_2]):
                res_to_matrix += el_1 * el_2
            sub_list.append(res_to_matrix)
            res_to_matrix = 0
        list_res.append(sub_list)
    return list_res

def number_on_matrix(num: int, matrix: list):
    matrix_ = [[elem * num for elem in row] for row in matrix]
    return matrix_
