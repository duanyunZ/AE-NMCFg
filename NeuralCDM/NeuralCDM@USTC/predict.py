import torch
import numpy as np
import json
import sys
import os
import warnings
from sklearn.metrics import roc_auc_score
from data_loader import ValTestDataLoader
from model import Net

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from SidePackage import auxiliary as aux
from SidePackage import evaluation as ev


def evaluate(epoch, source_path, dataset, prob_desc, R, q_m, student_n, exer_n, knowledge_n):
    data_loader = ValTestDataLoader(source_path, dataset, d_type='test')
    net = Net(student_n, exer_n, knowledge_n)
    device = torch.device('cpu')
    print('testing model...')
    data_loader.reset()
    load_snapshot(net, source_path + '/NeuralCDM/NeuralCDM@USTC/model_epoch/' + dataset + '/model_epoch' + str(epoch))
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    stu_kn_pro_dict = {}
    miss_coo = []

    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        out_put = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        out_put = out_put.view(-1)

        for _ in range(len(input_exer_ids)):
            exer = input_exer_ids[_].item()
            stu = input_stu_ids[_].item()
            miss_coo.append((exer, stu))

        stu_stat_emb = net.get_knowledge_status(input_stu_ids)
        for _ in range(len(input_stu_ids)):
            stu = input_stu_ids[_].item()
            stu_kn_pro_dict[stu] = stu_stat_emb[_].tolist()

        for i in range(len(input_exer_ids)):
            if prob_desc[input_exer_ids[i].item()] == 'Obj':
                exer_count += 1
                if (labels[i] == 1 and out_put[i] > 0.5) or (labels[i] == 0 and out_put[i] < 0.5):
                    correct_count += 1

        pred_all += out_put.tolist()
        label_all += labels.tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))

    # Build stu_kn_pro array (K, M)
    stu_kn_pro = np.zeros((knowledge_n, student_n))
    for stu, vec in stu_kn_pro_dict.items():
        stu_kn_pro[:, stu] = vec

    # Calculate KRC
    kn_krc_list = ev.cal_diag_krc(prob_desc, R, miss_coo, stu_kn_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])

    # Calculate PR (if prerequisite matrix exists)
    adj_matrix = None
    hier_path = source_path + '/NeuralCDM/NeuralCDM@USTC/data/' + dataset + '/hier_matrix.txt'
    if os.path.exists(hier_path):
        adj_matrix = np.loadtxt(hier_path, dtype=int)
        if adj_matrix.shape != (knowledge_n, knowledge_n):
            print(f"Warning: prerequisite matrix shape {adj_matrix.shape} != ({knowledge_n},{knowledge_n}), ignored.")
            adj_matrix = None

    if adj_matrix is not None and np.sum(adj_matrix) > 0:
        pr_005 = ev.cal_pr_value_05(stu_kn_pro, adj_matrix)
        pr_01 = ev.cal_pr_value_01(stu_kn_pro, adj_matrix)
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: %.5f, PR_0.1: %.5f" % (accuracy, rmse, krc, pr_005, pr_01))
    else:
        print("ACC: %.5f, RMSE: %.5f, KRC: %.5f, PR_0.05: N/A, PR_0.1: N/A" % (accuracy, rmse, krc))


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


if __name__ == '__main__':
    DATASET = input("\nplease choose a dataset: [FrcSub, Junyi-s, ednet, assist09, Quanlang-s, unit-bio-small, unit-his-small, unit-eng]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Junyi-s', 'ednet', 'assist09','Quanlang-s', 'unit-bio-small', 'unit-his-small', 'unit-eng']:
        warnings.warn("dataset does not exist.")
        exit()

    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        epoch = int(sys.argv[1])
    else:
        epoch = 20
        print(f"No epoch specified, using default epoch = {epoch}")

    with open(BASE_DIR + '/NeuralCDM/NeuralCDM@USTC/configs/' + DATASET + '/config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    R = np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/q.txt")
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(R.T)

    evaluate(epoch, BASE_DIR, DATASET, prob_desc, R.T, q_m, student_n, exer_n, knowledge_n)