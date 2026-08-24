# -*- coding: utf-8 -*-

import random
import numpy as np
from numpy.core.fromnumeric import shape


def missing_stu_exe(stu_exe, miss_rate):

    stu_num = stu_exe.shape[1]
    train_data = stu_exe.copy()

    test_loc = list()
    for stu in range(stu_num):
        nan_index = list(np.where(np.isnan(train_data[:,stu]))[0])
        del_num = int((len(train_data[:,stu]) - len(nan_index)) * miss_rate)
        _ = [i for i, j in enumerate(train_data[:,stu]) if i not in nan_index]  # the list that consists of index can be deleted
        random.shuffle(_)
        while del_num > 0 and len(_) > 0:
            del_index = _.pop()
            res_log = train_data[del_index,:].copy()  # copy the exercise response log
            res_log[stu] = np.NaN  # attemp to delete this record
            if np.isnan(res_log).all():
                # if the exercise (ID = del_index) has no anwser records
                # after delete this response log, ignore it.
                continue
            else:
                train_data[del_index][stu] = np.NaN
                test_loc.append([del_index, stu])
                del_num -= 1

    return train_data, test_loc


def matrix_miss_fill(stu_exe_miss):

    stu_exe_fill = stu_exe_miss.copy()
    # stu_exe_fill[np.isnan(stu_exe_fill)] = np.random.randint(0,2)
    stu_exe_fill[np.isnan(stu_exe_fill)] = 0

    return stu_exe_fill


def matrix_miss_fill_GBE(stu_exe_miss):
    shape = stu_exe_miss.shape
    stu_exe_fill = stu_exe_miss.copy()

    exe_avg = dict()
    stu_avg = dict()
    ans_avg = 0

    miss_coo = np.argwhere(np.isnan(stu_exe_fill)).tolist()
    ans_avg = stu_exe_fill[stu_exe_fill>=0].sum() / (shape[0] * shape[1] - len(miss_coo))

    for _ in range(shape[0]):
        exe_list = stu_exe_fill[_]
        ans_total = exe_list[exe_list>=0].sum()
        count = len(exe_list) - len(np.argwhere(np.isnan(exe_list)))
        exe_avg[_] = ans_total / count

    for _ in range(shape[1]):
        stu_list = stu_exe_fill[:,_]
        ans_total = stu_list[stu_list>=0].sum()
        count = len(stu_list)- len(np.argwhere(np.isnan(stu_list)))
        stu_avg[_] = ans_total / count

    for coo in miss_coo:
        stu_exe_fill[coo[0]][coo[1]] = exe_avg[coo[0]] + stu_avg[coo[1]] - ans_avg

    stu_exe_fill[stu_exe_fill < 0] = 0
    stu_exe_fill[np.isnan(stu_exe_fill)] = 0

    return stu_exe_fill


def stu_exe_repe_col(stu_exe, repe_rate):
    col = stu_exe.shape[1]
    repe_num = col * repe_rate
    count = 0
    rand_col_1 = np.random.randint(0, col)
    while count <= repe_num:
        rand_col_2 = np.random.randint(0, col)
        if rand_col_1 != rand_col_2:
            stu_exe[:, rand_col_2] = stu_exe[:, rand_col_1]

        count += 1

    return stu_exe


def stu_exe_repe_row(stu_exe, repe_rate):

    row = stu_exe.shape[0]
    repe_num = row * repe_rate
    count = 0
    rand_row_1 = np.random.randint(0, row)
    while count <= repe_num:
        rand_row_2 = np.random.randint(0, row)
        if rand_row_1 != rand_row_2:
            stu_exe[rand_row_2] = stu_exe[rand_row_1]

        count += 1

    return stu_exe