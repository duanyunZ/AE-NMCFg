import os
import re
import sys
import json
import numpy as np

DATASET = "Junyi-s"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
FILE_PATH = BASE_DIR + "/NeuralCDM@USTC/data/" + DATASET

prob_desc = {}

with open(FILE_PATH + '/log_data.json', encoding='utf8') as i_f:
    stus = json.load(i_f)
    for stu in stus:
        for log in stu['logs']:
            exer_id = log['exer_id']
            if log['score'] == 1.0 or log['score'] == 0.0:
                prob_desc[exer_id] = 'Obj'
            else:
                prob_desc[exer_id] = 'Sub'

_ = dict(sorted(prob_desc.items(), key=lambda x: x[0]))  # sorted
res = []
for exer_id, type in _.items():
    res.append([exer_id, type])

# save
with open(FILE_PATH+ "/problemdesc.txt", 'w') as file:
    np.savetxt(file, np.array(res) , delimiter=' ', fmt='%s')