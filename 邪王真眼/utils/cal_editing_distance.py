import numpy as np


def cal_editing_distance(string1, string2):
    matrix = np.zeros((len(string1) + 1, len(string2) + 1))
    for i in range(len(string1) + 1):
        matrix[i][0] = i
    for j in range(len(string2) + 1):
        matrix[0][j] = j
    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    edit_distance = matrix[len(string1)][len(string2)]
    return 1 - edit_distance / max(len(string1), len(string2))