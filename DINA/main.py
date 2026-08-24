import numpy as np
import json
import os
import logging
import warnings
import sys
import time

import DINA
import data_pre

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import auxiliary as aux
from SidePackage import evaluation as ev

DATASET = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
print("dataset %s is choosed" % DATASET)
if DATASET not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09','Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
    warnings.warn("dataset does not exist.")
    exit()

train_ratio = 0.8
valid_ratio = 0

for i in range(5):
    is_divide = "yes"
    if is_divide == "yes":
        data_pre.out_files(DATASET, train_ratio, valid_ratio)
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()

    FILE_PATH = BASE_DIR + "/DINA/data/" + DATASET

    q_m = np.loadtxt(FILE_PATH + "/q_m.csv", dtype=int, delimiter=',')
    prob_num, know_num = q_m.shape[0], q_m.shape[1]

    # Load training and test sets
    with open(FILE_PATH + "/train_data.json", encoding='utf-8') as file:
        train_set = json.load(file)

    with open(FILE_PATH + "/test_data.json", encoding='utf-8') as file:
        test_set = json.load(file)

    # Total number of students: covers both training and test sets
    max_train_id = max([x['user_id'] for x in train_set]) if train_set else -1
    max_test_id  = max([x['user_id'] for x in test_set])  if test_set  else -1
    stu_num = max(max_train_id, max_test_id) + 1

    # Build response matrix R (rows: students, columns: problems)
    R = -1 * np.ones(shape=(stu_num, prob_num))
    for log in train_set:
        R[log['user_id'], log['item_id']] = log['score']

    logging.getLogger().setLevel(logging.INFO)

    cdm = DINA.DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)

    start = time.time()
    cdm.train(epoch=20, epsilon=1e-3)
    end = time.time()
    print('TIME:%.5f' % (end - start))

    R_true = np.loadtxt(BASE_DIR + '/Data/' + DATASET + '/data.txt')
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(R.T)

    # Evaluate prediction accuracy and RMSE (KRC is manually calculated below)
    rmse, mae, acc, _ = cdm.eval(test_set, prob_desc, R_true.T)

    # Construct test set coordinate list
    miss_coo = [(log['item_id'], log['user_id']) for log in test_set]

    # Get student-knowledge proficiency matrix (K, M)
    stu_kn_pro = cdm.all_states[cdm.theta].T   # (know_num, stu_num)

    # Calculate KRC
    kn_krc_list = ev.cal_diag_krc(prob_desc, R_true.T, miss_coo, stu_kn_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values() if not np.isnan(x)])

    # Load prerequisite matrix and calculate PR
    adj_matrix = None
    hier_path = BASE_DIR + "/DINA/data/" + DATASET + "/hier_matrix.txt"
    if os.path.exists(hier_path):
        adj_matrix = np.loadtxt(hier_path, dtype=int)
        if adj_matrix.shape != (know_num, know_num):
            print(f"Warning: prerequisite matrix shape {adj_matrix.shape} != ({know_num},{know_num}), ignored.")
            adj_matrix = None

    if adj_matrix is not None and np.sum(adj_matrix) > 0:
        pr_005 = ev.cal_pr_value_05(stu_kn_pro, adj_matrix)
        pr_01  = ev.cal_pr_value_01(stu_kn_pro, adj_matrix)
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: %.5f, PR_0.1: %.5f" % (acc, rmse, krc, pr_005, pr_01))
    else:
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: N/A, PR_0.1: N/A" % (acc, rmse, krc))