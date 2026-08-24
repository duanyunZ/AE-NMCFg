import numpy as np
import math
import pandas as pd
import random
import warnings

import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import preprocessing as pre


def divide_dense(arr_data):
    exe_num = arr_data.shape[1]
    divide_ratio = [0.8, 0, 0.2]
    divide_num = [int(i * exe_num) for i in divide_ratio]

    arr_train = np.nan * np.ones(np.shape(arr_data))
    arr_vali = np.nan * np.ones(np.shape(arr_data))
    arr_test = np.nan * np.ones(np.shape(arr_data))
    arr_train_vali = np.nan * np.ones(np.shape(arr_data))

    for exm_count in range(arr_data.shape[0]):

        li_ran = [_ for _ in range(exe_num)]
        slice1 = random.sample(li_ran, divide_num[0])
        li_ran2 = list(set(li_ran) - set(slice1))
        slice2 = random.sample(li_ran2, divide_num[1])
        slice2 = slice2
        slice3 = list(set(li_ran2) - set(slice2))
        slice4 = slice1 + slice2

        for slice1_count in slice1:
            arr_train[exm_count][slice1_count] = arr_data[exm_count][slice1_count]
        for slice2_count in slice2:
            arr_vali[exm_count][slice2_count] = arr_data[exm_count][slice2_count]
        for slice3_count in slice3:
            arr_test[exm_count][slice3_count] = arr_data[exm_count][slice3_count]
        for slice4_count in slice4:
            arr_train_vali[exm_count][slice4_count] = arr_data[exm_count][slice4_count]

    np.savetxt(path + 'divide_data/train_data.txt', arr_train_vali)
    np.savetxt(path + 'divide_data/test_data.txt', arr_test)


def divide_sparse(arr_data):
    divide_ratio = [0.8, 0, 0.2]
    arr_train, test_loc = pre.missing_stu_exe(arr_data.T, divide_ratio[0])

    arr_test = np.nan * np.ones(np.shape(arr_data))
    for _ in test_loc:
        exe, stu = _[0], _[1]
        arr_test[stu][exe] = arr_data[stu][exe]

    np.savetxt(path + 'divide_data/train_data.txt', arr_train.T)
    np.savetxt(path + 'divide_data/test_data.txt', arr_test)


if __name__ == '__main__':

    DATA = input("\nplease choose a dataset: [FrcSub,Junyi-s,ednet,assist09,Quanlang-s,unit-bio-small,unit-his-small,unit-eng]: ")
    print("dataset %s is choosed" % DATA)
    if DATA not in ['FrcSub', 'ednet', 'Junyi-s', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()
    path = BASE_DIR + '/DeepIRT+/data/' + DATA + '/'

    is_divide = input("re-divide the dataset? (y or n): ")
    if is_divide == 'y':
        arr_data = np.loadtxt(path + 'data.txt')
        os.makedirs(path + 'divide_data/', exist_ok=True)
        if DATA in ['FrcSub']:
            divide_dense(arr_data)
        elif DATA in ['Junyi-s', 'ednet', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
            divide_sparse(arr_data)
        else:
            warnings.warn("dataset does not exist.")
            exit()

    train = np.loadtxt(path + 'divide_data/train_data.txt')
    os.makedirs(path + 'pkl_data/', exist_ok=True)
    dic_train = {}
    li_uid = []
    li_qid = []
    li_result = []
    for stu in range(train.shape[0]):
        for exe in range(train.shape[1]):
            if math.isnan(train[stu][exe]) is False:
                li_uid.append(str(stu))
                li_qid.append(str(exe))
                li_result.append(train[stu][exe])
    dic_train['userId'] = li_uid
    dic_train['topicId'] = li_qid
    dic_train['result'] = li_result
    df = pd.DataFrame(dic_train)
    df.to_pickle(path + 'pkl_data/train_uid_qid_res.pkl')

    test = np.loadtxt(path + 'divide_data/test_data.txt')
    dic_test = {}
    li_uid = []
    li_qid = []
    li_result = []
    for stu in range(test.shape[0]):
        for exe in range(test.shape[1]):
            if math.isnan(test[stu][exe]) is False:
                li_uid.append(str(stu))
                li_qid.append(str(exe))
                li_result.append(test[stu][exe])
    dic_test['userId'] = li_uid
    dic_test['topicId'] = li_qid
    dic_test['result'] = li_result
    df = pd.DataFrame(dic_test)
    df.to_pickle(path + 'pkl_data/test_uid_qid_res.pkl')