import os
from pickle import FALSE
import sys
import json
import numpy as np
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

DATASET = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
print("dataset %s is choosed" % DATASET)
if DATASET not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09', 'Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
    warnings.warn("dataset does not exist.")
    exit()

SAVE_PATH = BASE_DIR + "/NeuralCDM/NeuralCDM@USTC/data/" + DATASET

R = np.loadtxt(BASE_DIR + '/Data/' + DATASET + "/data.txt")  # loading the student-exercise matrix
Q = np.loadtxt(BASE_DIR + '/Data/' + DATASET + "/q.txt", dtype=int)  # loading the Q-Matrix

stu_num, prob_num = R.shape[0], R.shape[1]
knowledge_num = Q.shape[1]

config_dir = BASE_DIR + "/NeuralCDM/NeuralCDM@USTC/configs/" + DATASET
os.makedirs(config_dir, exist_ok=True)
with open(config_dir + "/config.txt", 'w') as file:
    np.savetxt(file, np.array([[stu_num, prob_num, knowledge_num]]), fmt='%d', delimiter=',',
               header='Number of Students, Number of Exercises, Number of Knowledge Concepts')

res = []
for stu in range(stu_num):
    # NOTE: all the index begin from 1, not 0
    has_prob_num = prob_num - len(np.argwhere(np.isnan(R[stu])).tolist())  # the answer number of stu
    stu_logs, logs = [], []
    for prob in range(prob_num):
        kn_set = [int(x[0])+1 for x in np.argwhere(Q[prob] == 1)]
        if not np.isnan(R[stu][prob]):
            log = {'exer_id': int(prob)+1, 'score': R[stu][prob], 'knowledge_code': kn_set}
            logs.append(log)
    stu_logs = [{'user_id': int(stu)+1, 'log_num': has_prob_num, 'logs': logs}]
    res += stu_logs

os.makedirs(SAVE_PATH, exist_ok=True)
with open(SAVE_PATH + "/log_data.json", 'w', encoding='utf8') as file:
    json.dump(res, file, indent=4, ensure_ascii=False)