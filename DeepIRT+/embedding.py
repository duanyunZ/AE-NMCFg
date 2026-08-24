import numpy as np
import os, sys
import pickle
import pandas as pd
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

DATA = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09,Quanlang-s,unit-bio-small,unit-his-small,unit-eng]: ")
print("dataset %s is choosed" % DATA)
if DATA not in ['FrcSub', 'ednet', 'Junyi-s', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
    warnings.warn("dataset does not exist.")
    exit()
path = BASE_DIR + '/DeepIRT+/data/' + DATA + '/'
emb_size = 50  # the length of the embedding (vector)

q_matrix=np.loadtxt(path + "q.txt")  # load Q-matrix

ex_num, kn_num = q_matrix.shape[0], q_matrix.shape[1]

# *** 1. generate qid_kcode_data.pkl ***
# Here we only consider the relationships between exercise (question) and knolwedge concepts
# --> exercise ID : {str(knowldge ID1), str(knowledge ID 2), ...}
qid_kcode = {}
for ex in range(ex_num):
    qid_kcode[str(ex)] = []
    li_kn = np.where(q_matrix[ex] == 1)[0].tolist()  # get the related knowledge concepts
    for kn in li_kn:
        qid_kcode[str(ex)].append(str(kn))

with open(path + "pkl_data/qid_kcode_data.pkl", "wb") as fp1: # Pickling
    pickle.dump(qid_kcode, fp1, protocol=pickle.HIGHEST_PROTOCOL)

# *** 2. generate qid_emb_data.pkl ***
# Here we randomly set the values for the embedding
qid_emb = {}
for ex in range(ex_num):
    num = len(np.where(q_matrix[ex] == 1)[0].tolist())  # get the number of the related knowledge concepts
    arr = np.random.random((num, emb_size))
    qid_emb[str(ex)] = arr
with open(path + "pkl_data/qid_emb_data.pkl", "wb") as fp2: # Pickling
    pickle.dump(qid_emb, fp2, protocol=pickle.HIGHEST_PROTOCOL)

# *** 3. generate code_emb_data.pkl ***
# Here we randomly set the values for the embedding
kcode_emb = {}
for kn in range(kn_num):
    arr = np.random.random((emb_size))
    kcode_emb[str(kn)] = arr
df=pd.DataFrame(kcode_emb)
df=df.T
df.to_pickle(path + '/pkl_data/code_emb_data.pkl')