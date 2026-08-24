import json
import numpy as np
import random
import os, sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import auxiliary as aux


def read_probdesc(filepath):
    prob_type = np.loadtxt(filepath, dtype=int, delimiter='\t',
                           converters={0: lambda x: int(x) - 1, 1: lambda s: s == b'Obj'}, skiprows=1)
    obj_prob_index, sub_prob_index = prob_type[prob_type[:, 1] == 1][:, 0], prob_type[prob_type[:, 1] == 0][:, 0]
    return prob_type, obj_prob_index, sub_prob_index

def out_files(dataset, train_ratio, valid_ratio):
    """
    output:
    1. obj_prob_index.csv
    2. sub_prob_index.csv
    3. q_m.csv
    4. train_data.json
    5. valid_data.json
    6. test_data.json
    """

    np.savetxt(BASE_DIR+"/DINA/data/"+dataset+"/q_m.csv", 
    np.loadtxt(BASE_DIR+"/Data/"+dataset+"/q.txt", dtype=int), delimiter=',', fmt='%d')

    R = np.loadtxt(BASE_DIR + "/Data/" + dataset + "/data.txt")
    stu_num, prob_num = R.shape[0], R.shape[1]
    train_logs, valid_logs, test_logs = [], [], []
   
    for stu in range(stu_num):
        if len(np.argwhere(np.isnan(R[stu]))) == prob_num:
            continue
        stu_logs = []
        _ = 0
        for prob in range(prob_num):
            if np.isnan(R[stu][prob]):
                continue
            log = {'user_id': int(stu), 'item_id': int(prob), 'score': R[stu][prob]}
            stu_logs.append(log)
            _ += 1
        random.shuffle(stu_logs)
        train_logs += stu_logs[: int(train_ratio * _)]
        valid_logs += stu_logs[int(train_ratio * _): int(train_ratio * _) + int(valid_ratio * _)]
        test_logs += stu_logs[int(train_ratio * _) + int(valid_ratio * _):]
    
    if os.path.exists(BASE_DIR + "/Data/" + dataset + "/problemdesc.txt"):
        file = BASE_DIR + "/Data/" + dataset + "/problemdesc.txt"
        problems, obj_prob_index, sub_prob_index = read_probdesc(file)
        np.savetxt(BASE_DIR+"/DINA/data/" + dataset + "/obj_prob_index.csv", obj_prob_index, delimiter=',', fmt='%d')
        np.savetxt(BASE_DIR+"/DINA/data/" + dataset + "/sub_prob_index.csv", sub_prob_index, delimiter=',', fmt='%d')
    else:
        prob_desc = aux.build_problem_desc(R.T)
        obj_prob_index = [[key] for key, value in prob_desc.items() if value == 'Obj']
        sub_prob_index = [[key] for key, value in prob_desc.items() if value == 'Sub']
        with open(BASE_DIR+"/DINA/data/" + dataset + "/obj_prob_index.csv", 'w') as file:
              writer = csv.writer(file)
              writer.writerows(obj_prob_index)
        with open(BASE_DIR+"/DINA/data/" + dataset + "/sub_prob_index.csv", 'w') as file:
              writer = csv.writer(file)
              writer.writerows(sub_prob_index)

    with open(BASE_DIR+"/DINA/data/" + dataset + "/train_data.json", 'w', encoding='utf8') as file:
              json.dump(train_logs, file, indent=4, ensure_ascii=False)

    with open(BASE_DIR+"/DINA/data/" + dataset + "/valid_data.json", 'w', encoding='utf8') as file:
              json.dump(valid_logs, file, indent=4, ensure_ascii=False)

    with open(BASE_DIR+"/DINA/data/" + dataset + "/test_data.json", 'w', encoding='utf8') as file:
              json.dump(test_logs, file, indent=4, ensure_ascii=False)

    
