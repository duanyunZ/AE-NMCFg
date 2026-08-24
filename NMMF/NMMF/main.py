# -*- coding: utf-8 -*-
"""
The NMMF model: Non-Negative Multiple Matrix Factorization
Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence (IJCAI, 2013)
"""

import os
import sys
import warnings
import numpy as np
import nmmf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux


def train_test(stu_exe, stu_exe_fill, q_m, test_loc, prob_desc, rank, beta, dataset):
    """
    Training and testing
    """
    # --- training ---
    b, w, h = nmmf.fit_data(stu_exe_fill, q_m, rank, beta)
    stu_exe_pre = np.dot(w, h)

    # --- testing ---
    accuracy_obj = ev.cal_accuracy_obj(stu_exe, stu_exe_pre, test_loc, prob_desc)
    rmse = ev.cal_rmse(stu_exe, stu_exe_pre, test_loc)

    # Student-knowledge proficiency matrix (K, M)
    # b shape is (rank, K), transpose to (K, rank)
    stu_kn_pro = np.dot(b.T, h)   # (K, rank) @ (rank, M) => (K, M)

    # KRC calculation (convert to dictionary keyed by knowledge)
    stu_kn_pro_dict = {}
    for kn in range(stu_kn_pro.shape[0]):
        stu_kn_pro_dict[kn] = stu_kn_pro[kn, :].tolist()
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_kn_pro_dict, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])

    # PR calculation (prerequisite matrix file: hier_matrix@dataset.txt)
    adj_matrix = None
    hier_path = BASE_DIR + '/NMMF/NMMF/data/' + '/hier_matrix@' + dataset + '.txt'
    if os.path.exists(hier_path):
        adj_matrix = np.loadtxt(hier_path, dtype=int)
        if adj_matrix.shape != (q_m.shape[1], q_m.shape[1]):
            print(f"Warning: prerequisite matrix shape {adj_matrix.shape} != ({q_m.shape[1]},{q_m.shape[1]}), ignored.")
            adj_matrix = None

    if adj_matrix is not None and np.sum(adj_matrix) > 0:
        pr_005 = ev.cal_pr_value_05(stu_kn_pro, adj_matrix)
        pr_01 = ev.cal_pr_value_01(stu_kn_pro, adj_matrix)
        print("ACCURACY (Obj): %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: %.5f, PR_0.1: %.5f"
              % (accuracy_obj, rmse, krc, pr_005, pr_01))
    else:
        print("ACCURACY (Obj): %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: N/A, PR_0.1: N/A"
              % (accuracy_obj, rmse, krc))


if __name__ == '__main__':
    DATASET = input("\nplease choose a dataset: [FrcSub, Math1, Math2, Quanlang, A0910, Quanlang-s, Junyi, Junyi-s, unit-eng, unit-bio-small, unit-his-small, assist09,ednet]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small','assist09', 'ednet']:
        warnings.warn("dataset does not exist.")
        exit()

    MISS_R = 0.2

    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/q.txt", dtype=int)

    is_divide = "yes"
    if is_divide == "yes":
        train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
        os.makedirs(BASE_DIR + "/NMMF/data/", exist_ok=True)
        np.savetxt(BASE_DIR + "/NMMF/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
        np.savetxt(BASE_DIR + "/NMMF/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ', fmt='%s')
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()

    train_data = ((np.loadtxt(BASE_DIR + "/NMMF/data/train@" + DATASET + ".txt")).astype(float))
    test_loc = aux.read_test_loc(BASE_DIR + "/NMMF/data/test@" + DATASET + ".txt")

    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)

    stu_exe_fill = pre.matrix_miss_fill(train_data)
    rank = 3
    beta = 0.5
    train_test(stu_exe, stu_exe_fill, q_m, test_loc, prob_desc, rank, beta, DATASET)