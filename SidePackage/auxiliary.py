# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la

def read_problem_desc(filename):
    exe_desc = dict()
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            desc = [i for i in lines.split()]
            if len(desc) > 3:
                continue
            exe_desc[int(desc[0])-1] = desc[1]

    return exe_desc


def build_problem_desc(stu_exe):
    exe_num = stu_exe.shape[0]
    problem_description = dict()
    for exe_id in range(exe_num):
        problem_description[exe_id] = 'Obj'
    sub_coo = np.argwhere((stu_exe > 0) & (stu_exe < 1)).tolist()
    for coo in sub_coo:
        problem_description[coo[0]] = 'Sub'

    return problem_description


def read_test_loc(filename):
    _ = list()
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            _.append([int(i) for i in lines.split()])

    test_loc = list()
    for coo in _:
        test_loc.append(tuple(coo))

    return test_loc


def get_rank(x_matrix):
    left_u, sigma, right_vt = la.svd(x_matrix)
    sin_deg = 0
    print(sigma)
    for _ in sigma:
        sin_deg += _
    print(sigma[0] / sin_deg)