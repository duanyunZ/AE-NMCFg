# -*- coding: utf-8 -*-
"""
SNMCF model training and testing

"""

import os
import sys
import warnings
import numpy as np
import snmcf
import time
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux


def snmcf_train(train_data, train_fill, q_m, rank, alpha, lu, lv, le, gbe=False):
    print("\nSNMCF hyperparameters: \nrank: %d\nAlpha: %.2f\nLambda U: %.2f\nLambda V: %.2f\nLambda E: %.2f"
          % (rank, alpha, lu, lv, le))

    if gbe is True:
        w = snmcf.cal_weight_matrix(train_fill)
    else:
        w = snmcf.cal_weight_matrix(train_data)

    start = time.time()
    u, v, e = snmcf.fit_data_mult(train_fill, q_m, w, rank, alpha, lu, lv, le)
    stu_pro = snmcf.stu_kn_diagnose_area(u, v)  # shape (M, K)
    stu_exe_pre = snmcf.cal_matrix_pre(left_latent_matrix=e, right_latent_matrix=u)
    q_pre = snmcf.cal_matrix_pre(left_latent_matrix=e, right_latent_matrix=v)
    end = time.time()
    print('TIME:%.5f' %(end - start))
    return stu_pro, stu_exe_pre, q_pre


def snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, adj_matrix, test_loc, prob_desc, cl):
    # Transpose to (K, M)
    stu_pro_KM = stu_pro.T

    # Calculate basic metrics
    accuracy_obj = ev.cal_accuracy_obj(stu_exe, stu_exe_pre, test_loc, prob_desc)
    rmse = ev.cal_rmse(stu_exe, stu_exe_pre, test_loc)

    # KRC
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_pro_KM, q_m)
    krc = np.mean([x for x in kn_krc_list.values() if not np.isnan(x)])

    # Calculate PR (if prerequisite matrix exists)
    if adj_matrix is not None and np.sum(adj_matrix) > 0:
        pr_005 = ev.cal_pr_value_05(stu_pro_KM, adj_matrix)
        pr_01  = ev.cal_pr_value_01(stu_pro_KM, adj_matrix)
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: %.5f, PR_0.1: %.5f"
              % (accuracy_obj, rmse, krc, pr_005, pr_01))
    else:
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: N/A, PR_0.1: N/A (no prerequisite matrix)"
              % (accuracy_obj, rmse, krc))


if __name__ == '__main__':
    MISS_R = 0.2
    RANK = 1
    ALPHA = 1
    LE = 0.01
    LU = 1
    LV = 5
    is_GBE = False
    CL = 0.05

    DATASET = input("\nplease choose a dataset: [FrcSub, ednet, assist09, Junyi-s, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'ednet', 'assist09', 'Junyi-s','Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()

    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET +"/q.txt", dtype=int)
    kn_num = q_m.shape[1]

    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)

    # Load prerequisite matrix (directly read txt file)
    adj_matrix = None
    hier_path = BASE_DIR + "/SNMCF/data/hier_matrix@" + DATASET + ".txt"
    if os.path.exists(hier_path):
        adj_matrix = np.loadtxt(hier_path, dtype=int)
        # Ensure shape is K x K
        if adj_matrix.shape != (kn_num, kn_num):
            print(f"Warning: prerequisite matrix shape {adj_matrix.shape} != ({kn_num},{kn_num}), ignored.")
            adj_matrix = None

    is_divide = input("re-divide the dataset? (yes or no): ")
    if is_divide == "yes":
        train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
        np.savetxt(BASE_DIR + "/SNMCF/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
        np.savetxt(BASE_DIR + "/SNMCF/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ', fmt='%s')
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()

    train_data = ((np.loadtxt(BASE_DIR + "/SNMCF/data/train@" + DATASET + ".txt")).astype(float))
    test_loc = aux.read_test_loc(BASE_DIR + "/SNMCF/data/test@" + DATASET + ".txt")

    if DATASET in ['FrcSub']:
        train_fill = pre.matrix_miss_fill_GBE(train_data)
        is_GBE = True
    elif DATASET in ['assist09', 'ednet', 'Junyi-s','Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        train_fill = pre.matrix_miss_fill(train_data)

    stu_pro, stu_exe_pre, q_pre = snmcf_train(train_data, train_fill, q_m, RANK, ALPHA, LU, LV, LE, gbe=is_GBE)
    snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, adj_matrix, test_loc, prob_desc, CL)