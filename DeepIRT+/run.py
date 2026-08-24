from moduel.LSTM import LSTM
from data.loader import DataLoader
from model.DeepModel import DeepIRT
import numpy as np
import pickle
import torch.optim as optim
import eval
import warnings
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import evaluation as ev

lstm = LSTM(50, 5)


def preparing(path):
    train_uid_qid_res = pickle.load(open(path + '/pkl_data/train_uid_qid_res.pkl', 'rb'))
    test_uid_qid_res = pickle.load(open(path + '/pkl_data/test_uid_qid_res.pkl', 'rb'))
    qid_emb = pickle.load(open(path + '/pkl_data/qid_emb_data.pkl', 'rb'))
    kcode_emb = pickle.load(open(path + '/pkl_data/code_emb_data.pkl', 'rb'))
    qid_kcode = pickle.load(open(path + '/pkl_data/qid_kcode_data.pkl', 'rb'))

    trainLoader, testLoader, uid_index_dict, qid_index_dict, kcode_index_dict \
        = DataLoader(train_uid_qid_res, test_uid_qid_res, qid_emb, kcode_emb, qid_kcode)
    kn_num = len(kcode_emb)
    st_num = len(set(train_uid_qid_res.loc[:, 'userId'].tolist()))

    return trainLoader, testLoader, kn_num, st_num, uid_index_dict, qid_index_dict, kcode_index_dict, test_uid_qid_res


def load_prerequisite_matrix(dataset_name, base_dir):
    filepath = os.path.join(base_dir, 'DeepIRT+/data', dataset_name, 'hier_matrix.txt')
    if os.path.exists(filepath):
        print("Loading prerequisite matrix from:", filepath)
        return np.loadtxt(filepath, dtype=int)
    else:
        warnings.warn("hier_matrix.txt not found, using zero matrix. PR will be 0.")
        return None


if __name__ == '__main__':

    hyper1 = 30
    print('dnn_hidden_size:', hyper1)

    DATA = input("\nplease choose a dataset: [FrcSub, Junyi-s, assist09, ednet, Quanlang-s,unit-bio-small,unit-his-small,unit-eng]: ")
    print("dataset %s is choosed" % DATA)
    if DATA not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()
    path = BASE_DIR + '/DeepIRT+/data/' + DATA + '/'

    if os.path.exists(path + 'problemdesc.txt'):
        with open(path + 'problemdesc.txt', 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        prob_desc = []
        for line in lines:
            if line in ['0', '1']:
                prob_desc.append('Obj' if line == '0' else 'Sub')
            else:
                prob_desc.append(line)
    else:
        arr_data = np.loadtxt(path + 'data.txt')
        from SidePackage import auxiliary as aux
        prob_desc = aux.build_problem_desc(arr_data.T)

    q_matrix = np.loadtxt(path + 'q.txt')
    arr_data = np.loadtxt(path + 'data.txt')

    train, test, kn_num, st_num, uid_index_dict, qid_index_dict, kcode_index_dict, test_uid_qid_res = preparing(path)

    start = time.time()
    model = DeepIRT(lstm_input_size=50, lstm_hidden_size=1, dnn_input_size=50, dnn_hidden_size=hyper1,
                    dnn_output_size=1, knowledge_nums=kn_num, student_nums=st_num, denseDim=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = eval.myLoss()
    model, optimizer = eval.train_epoch(model, train, optimizer, loss_func)

    acc_obj, rmse, mae, stu_kn_pro = eval.test_epoch(DATA, model, test, loss_func,
                                                     uid_index_dict, kcode_index_dict, prob_desc, q_matrix)

    krc = np.nan
    if stu_kn_pro is not None and len(test_uid_qid_res) > 0:
        stu_kn_pro_T = stu_kn_pro.T

        all_pro = []
        all_ans = []
        for idx in range(len(test_uid_qid_res)):
            u_idx = int(test_uid_qid_res.iloc[idx]['userId'])
            i_idx = int(test_uid_qid_res.iloc[idx]['topicId'])
            score = int(test_uid_qid_res.iloc[idx]['result'])

            q_row = q_matrix[i_idx]
            kn_indices = np.where(q_row == 1)[0]
            if len(kn_indices) > 0:
                pro = np.mean([stu_kn_pro_T[kn, u_idx] for kn in kn_indices])
                all_pro.append(pro)
                all_ans.append(score)

        print(f"[DEBUG] valid samples: pro={len(all_pro)}, ans={len(all_ans)}")
        if len(all_pro) > 0:
            pairs = list(zip(all_pro, all_ans))
            num_pos = sum(1 for p, a in pairs if a == 1)
            num_neg = sum(1 for p, a in pairs if a == 0)
            print(f"[DEBUG] pos={num_pos}, neg={num_neg}")
            if num_pos > 0 and num_neg > 0:
                krc = ev.__cal_binary_krc(pairs, label=[0, 1])
                print('KRC:', krc)
            else:
                print(f"KRC: NaN, lack pos/neg sample, pos={num_pos}, neg={num_neg}")
        else:
            print('KRC: NaN (no valid test samples)')
    else:
        print('KRC: NaN (stu_kn_pro is None or test set empty)')

    pr_005 = np.nan
    pr_01 = np.nan
    if stu_kn_pro is not None:
        stu_kn_pro_T = stu_kn_pro.T
        adj = load_prerequisite_matrix(DATA, BASE_DIR)
        if adj is None:
            kn_num = q_matrix.shape[1]
            adj = np.zeros((kn_num, kn_num), dtype=int)

        pr_005 = ev.cal_pr_value(stu_kn_pro_T, adj, alpha=0.05)
        pr_01 = ev.cal_pr_value(stu_kn_pro_T, adj, alpha=0.1)
        print('PR_0.05:', pr_005)
        print('PR_0.1 :', pr_01)
    else:
        print('PR_0.05: None (stu_kn_pro is None)')
        print('PR_0.1 : None')

    print('acc_obj', acc_obj)
    print('rmse:', rmse)
    end = time.time()
    print('TIME:%.5f' % (end - start))