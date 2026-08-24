import json
import random
import os
import sys
import warnings

# ===== BASE_DIR =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# =================================

min_log = 3


def divide_data(filepath, train_ratio, test_ratio):
    os.makedirs(filepath, exist_ok=True)

    with open(filepath + '/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    stu_i = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < min_log:
            del stus[stu_i]
            stu_i -= 1
        stu_i += 1

    # 2. 8:2
    train_set, test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        logs = stu['logs'][:]
        random.shuffle(logs)

        train_size = int(stu['log_num'] * train_ratio)
        test_size = stu['log_num'] - train_size

        stu_train = {'user_id': user_id, 'log_num': train_size, 'logs': logs[:train_size]}
        stu_test = {'user_id': user_id, 'log_num': test_size, 'logs': logs[train_size:]}

        train_set.append(stu_train)
        test_set.append(stu_test)

    train_flat = []
    for stu in train_set:
        for log in stu['logs']:
            train_flat.append({
                'user_id': stu['user_id'],
                'exer_id': log['exer_id'],
                'score': log['score'],
                'knowledge_code': log['knowledge_code']
            })
    random.shuffle(train_flat)

    with open(filepath + '/train_set.json', 'w', encoding='utf8') as f:
        json.dump(train_flat, f, indent=4, ensure_ascii=False)
    with open(filepath + '/test_set.json', 'w', encoding='utf8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    DATASET = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09', 'Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()

    FILE_PATH = BASE_DIR + "/NeuralCDM/NeuralCDM@USTC/data/" + DATASET
    divide_data(FILE_PATH, 0.8, 0.2)