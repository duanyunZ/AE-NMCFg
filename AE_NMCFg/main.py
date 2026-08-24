# -*- coding: utf-8 -*-
"""
AE-NMCFg model training and testing.
"""

import os
import sys
import warnings
import time
import numpy as np
import pickle
import logging
import math
from scipy import integrate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import ae_nmcfg
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux
from SidePackage import evaluation as ev


def normal(x):
    """Standard normal probability density function."""
    return 1 / math.sqrt(2 * math.pi) * math.exp(-math.pow(x, 2) / 2)


def cal_weight(student_exe):
    """Binary mask: 1 for observed, 0 for missing."""
    weight_matrix = np.ones(shape=student_exe.shape)
    weight_matrix[np.isnan(student_exe)] = 0
    return weight_matrix.astype(np.int64)


def cal_stu_ex_pre(q_m, u, v, m, st_num):
    """Predict response probability via probit: Φ(Q·V^T·U + mu)."""
    phi_fun = np.frompyfunc(lambda x: integrate.quad(normal, -float('inf'), x)[0], 1, 1)
    delta = np.dot(np.dot(q_m, v.T), u) + np.tile(m, st_num)
    phi = phi_fun(delta)
    return phi


def gat_aenmcf_train(train_data, train_fill, q_m, adj_matrix, rank,
                     lambda_reg, beta_reg, step_search, gbe=False, max_iter=500, cri=1e-2,
                     early_stop_delta_threshold=None,
                     early_stop_patience=10,
                     early_stop_patience_no_improve=50,
                     early_stop_enabled=True):
    """Train AE-NMCFg on the given dataset."""
    if early_stop_delta_threshold is None:
        early_stop_delta_threshold = cri

    print("\nAE-NMCF hyperparameters:")
    print("rank: %d, lambda: %.3f, beta: %.4f" % (rank, lambda_reg, beta_reg))
    print("max_iter: %d, convergence_threshold: %.1e" % (max_iter, cri))
    print("early_stop_delta_threshold: %.1e" % early_stop_delta_threshold)

    # Construct binary mask
    if gbe is True:
        w = cal_weight(train_fill)
    else:
        w = cal_weight(train_data)

    start = time.time()
    model = ae_nmcfg.ae_nmcfg(train_data, train_fill, q_m, w, rank,
                              adj_matrix, lambda_reg, beta_reg)
    u, e, v, m, r_matrix = model.train(max_iter=max_iter, cri=cri,
                                       early_stop_delta_threshold=early_stop_delta_threshold,
                                       early_stop_patience=early_stop_patience,
                                       early_stop_patience_no_improve=early_stop_patience_no_improve,
                                       early_stop_enabled=early_stop_enabled)
    end = time.time()
    print('TIME:%.5f' % (end - start))

    return u, e, v, m, r_matrix


def gat_aenmcf_test(stu_exe, test_loc, q_m, know_graph, u, e, v, m, prob_desc, cl, adj_matrix):
    """Evaluate trained AE-NMCFg: ACC, RMSE, KRC, PR_0.05, PR_0.1."""
    ex_num, st_num = stu_exe.shape
    kn_num = q_m.shape[1]

    # A = V^T U: knowledge proficiency matrix (K x M)
    stu_kn_pro = np.dot(v.T, u)

    # Normalize proficiency scores to [0, 1] for visualization (optional)
    stu_kn_pro_norm = np.zeros(shape=(stu_kn_pro.shape), dtype=float)
    for kn in range(kn_num):
        row = stu_kn_pro[kn, :]
        max_val = row.max()
        min_val = row.min()
        if max_val > min_val:
            stu_kn_pro_norm[kn, :] = (row - min_val) / (max_val - min_val)
        else:
            stu_kn_pro_norm[kn, :] = row

    # Predict response probabilities via probit link
    stu_ex_pre = cal_stu_ex_pre(q_m, u, v, m, st_num)

    # Prediction metrics
    accuracy_obj = ev.cal_accuracy_obj(stu_exe, stu_ex_pre, test_loc, prob_desc)
    rmse = ev.cal_rmse(stu_exe, stu_ex_pre, test_loc)

    # Cognitive diagnosis metrics
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_kn_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])

    # Prerequisite satisfaction rates
    pr = ev.cal_pr_value_05(stu_kn_pro, adj_matrix)
    pr_1 = ev.cal_pr_value_01(stu_kn_pro, adj_matrix)

    print("ACCURACY: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: %.5f, PR_0.1: %.5f" % (accuracy_obj, rmse, krc, pr, pr_1))
    return accuracy_obj, rmse, krc, pr, pr_1


def load_prerequisite_matrix(dataset_name):
    """Load prerequisite graph adjacency. Returns identity matrix if not found."""
    filepath = BASE_DIR + "/AE_NMCFg/data/hier_matrix@" + dataset_name + ".txt"
    if os.path.exists(filepath):
        print("Loading prerequisite matrix from: %s" % filepath)
        return np.loadtxt(filepath, dtype=int)
    else:
        print("Warning: Prerequisite matrix not found, using identity matrix")
        q_path = BASE_DIR + "/Data/" + dataset_name + "/q.txt"
        if os.path.exists(q_path):
            q_m = np.loadtxt(q_path, dtype=int)
            kn_num = q_m.shape[1]
        else:
            kn_num = 10
        return np.eye(kn_num)


def write_train_log(dataset, rank, lambda_reg, beta_reg, acc, rmse, krc, pr, pr_1):
    """Append training results to log file."""
    log_path = os.path.join(BASE_DIR, "AE_NMCFg", "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("dataset,rank,lambda,beta,ACC,RMSE,KRC,PR_0.05,PR_0.1\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{dataset},{rank},{lambda_reg},{beta_reg},{acc:.5f},{rmse:.5f},{krc:.5f},{pr:.5f},{pr_1:.5f}\n")


if __name__ == '__main__':

    MISS_R = 0.2
    is_GBE = False
    CL = 0.05
    lambda_reg = 0.06
    RANK = 4
    beta_reg = 0.005
    MAX_ITER = 500
    CRI = 1

    DATASET = input(
        "\nplease choose a dataset: [FrcSub, Junyi-s, assist09, ednet, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
    print("dataset %s is choosed" % DATASET)

    if DATASET not in ['FrcSub', 'Junyi-s', 'assist09', 'ednet', 'Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()

    # Load data (transposed: exercises x students)
    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    non_nan_index = np.where(~np.isnan(stu_exe))
    stu_exe[non_nan_index] = stu_exe[non_nan_index].astype(int)

    # Load Q-matrix
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/q.txt", dtype=int)

    # Load problem descriptions
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)

    # Load prerequisite graph
    kn_graph = None
    adj_matrix = load_prerequisite_matrix(DATASET)

    step_search = 'lipschitz'
    print('The step-size searching method is: %s' % step_search)

    # 10 repeated trials
    for i in range(10):
        is_divide = input("re-divide the dataset? (yes or no): ")
        if is_divide == "yes":
            # Split data into train (80%) and test (20%)
            train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
            os.makedirs(BASE_DIR + "/AE_NMCFg/data/", exist_ok=True)
            np.savetxt(BASE_DIR + "/AE_NMCFg/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
            np.savetxt(BASE_DIR + "/AE_NMCFg/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ',
                       fmt='%s')
            print("the data division has been completed.")
        elif is_divide == "no":
            train_data = (np.loadtxt(BASE_DIR + "/AE_NMCFg/data/train@" + DATASET + ".txt")).astype(float)
            test_loc = aux.read_test_loc(BASE_DIR + "/AE_NMCFg/data/test@" + DATASET + ".txt")
        else:
            warnings.warn("illegal input!")
            exit()

        if is_divide == "yes":
            train_data = (np.loadtxt(BASE_DIR + "/AE_NMCFg/data/train@" + DATASET + ".txt")).astype(float)
            test_loc = aux.read_test_loc(BASE_DIR + "/AE_NMCFg/data/test@" + DATASET + ".txt")

        # Fill missing values in training data
        if DATASET in ['FrcSub']:
            train_fill = pre.matrix_miss_fill_GBE(train_data)
            is_GBE = True
        elif DATASET in ['Junyi-s', 'assist09', 'ednet', 'Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
            train_fill = pre.matrix_miss_fill(train_data)

        u, e, v, m, r_matrix = gat_aenmcf_train(
            train_data, train_fill, q_m, adj_matrix, RANK,
            lambda_reg, beta_reg, step_search, gbe=is_GBE,
            max_iter=MAX_ITER, cri=CRI)

        acc, rmse, krc, pr, pr_1 = gat_aenmcf_test(
            stu_exe, test_loc, q_m, kn_graph, u, e, v, m, prob_desc, CL, adj_matrix
        )

        write_train_log(DATASET, RANK, lambda_reg, beta_reg, acc, rmse, krc, pr, pr_1)
        # print("\nLearned prerequisite weight matrix (R):")
        # print(np.round(r_matrix, 3))