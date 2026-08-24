import argparse
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from data_loader import TestDataLoader
from model import Net
import warnings
import json
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')

# Import KRC calculation functions
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Read directed graph
index_d_src, index_d_dst = [], []
d_edge_n = 0
with open("data/ednet/direct_graph.txt", 'r') as file:
    for line in file:
        src, dst = map(int, line.strip().split())
        d_edge_n += 1
        index_d_src.append(src)
        index_d_dst.append(dst)
edge_index_directed = torch.tensor([index_d_src, index_d_dst], dtype=torch.long)


# PR calculation functions
def cal_pr_value(stu_kn_pro, adj_matrix, alpha=0.05):
    M, K = stu_kn_pro.shape
    p_values = []
    edges = np.argwhere(adj_matrix == 1)
    if len(edges) == 0:
        return 0.0
    for m in range(M):
        differences = []
        for i, j in edges:
            diff = stu_kn_pro[m, i] - stu_kn_pro[m, j]
            differences.append(diff)
        differences = np.array(differences)
        if np.all(np.abs(differences) < 1e-10):
            p_value = 1.0
        else:
            try:
                _, p_value = wilcoxon(differences, alternative='greater')
            except:
                p_value = 1.0
        p_values.append(p_value)
    pr = np.mean(np.array(p_values) < alpha)
    return pr


def cal_pr_value_05(stu_kn_pro, adj_matrix):
    return cal_pr_value(stu_kn_pro, adj_matrix, alpha=0.05)


def cal_pr_value_01(stu_kn_pro, adj_matrix):
    return cal_pr_value(stu_kn_pro, adj_matrix, alpha=0.1)


# KRC calculation functions
def calculate_krc(stu_exe, test_data, stu_kn_pro_aligned, Q_matrix, prob_desc):
    # stu_kn_pro_aligned: [knowledge_n, n_user] aligned to actual user_id

    miss_coo = []
    for idx in range(len(test_data)):
        user_id = int(test_data.iloc[idx]['user_id'])
        item_id = int(test_data.iloc[idx]['exer_id'])
        miss_coo.append((item_id, user_id))  # Use actual user_id directly

    kn_krc_list = ev.cal_diag_krc(
        prob_desc,
        stu_exe,
        miss_coo,
        stu_kn_pro_aligned,  # [knowledge_n, n_user]
        Q_matrix
    )

    valid_krc = [v for v in kn_krc_list.values() if not np.isnan(v)]
    if len(valid_krc) > 0:
        return np.mean(valid_krc)
    else:
        return np.nan


# Build stu_exe from log_data.json
def build_stu_exe(log_data_path, n_user, n_item):
    with open(log_data_path, encoding='utf8') as f:
        stus = json.load(f)

    stu_exe = np.full((n_item, n_user), np.nan)

    for stu in stus:
        user_id = stu['user_id']
        for log in stu['logs']:
            exer_id = log['exer_id']
            score = log['score']
            if user_id < n_user and exer_id < n_item:
                stu_exe[exer_id, user_id] = score

    return stu_exe


def load_snapshot(model, filename, device):
    with open(filename, 'rb') as f:
        state = torch.load(f, map_location=device)
    model.load_state_dict(state)


def tes(args):
    # Load Q matrix
    Q_matrix = np.loadtxt("data/ednet/q.txt", dtype=int)
    # Build stu_exe from log_data.json
    stu_exe = build_stu_exe("data/ednet/log_data.json", args.student_n, args.exer_n)
    # prob_desc: all exercises default to objective
    prob_desc = ['Obj'] * Q_matrix.shape[0]

    with open('data/ednet/test_set.json', 'r', encoding='utf8') as f:
        test_logs = json.load(f)
    test_df = pd.DataFrame(test_logs)
    test_df.rename(columns={'user_id': 'user_id', 'exer_id': 'exer_id', 'score': 'score'}, inplace=True)

    knowledge_n = args.knowledge_n
    adj_matrix = np.zeros((knowledge_n, knowledge_n))
    with open("data/ednet/direct_graph.txt", 'r') as file:
        for line in file:
            src, dst = map(int, line.strip().split())
            adj_matrix[src, dst] = 1

    data_loader = TestDataLoader()
    net = Net(args).to(device)
    print('Loading model...')
    load_snapshot(net, 'model/best_model.pt', device=device)
    net.eval()

    all_labels = []
    all_preds = []
    all_knowledge = []
    all_user_ids = []  # Record user_id

    ei_d = edge_index_directed.to(device)

    with torch.no_grad():
        while not data_loader.is_end():
            stu, exer, qn, lab, qd = data_loader.next_batch()
            if stu is None:
                break

            stu = stu.to(device)
            exer = exer.to(device)
            qn = qn.to(device)
            lab = lab.to(device)
            qd = qd.to(device)

            logits, _, stu_knowledge_pro = net(stu, exer, qn, qd, ei_d)
            probs = torch.sigmoid(logits.view(-1))

            all_labels.extend(lab.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())
            all_knowledge.append(stu_knowledge_pro.cpu().numpy())
            all_user_ids.extend(stu.cpu().numpy())

    stu_knowledge_matrix = np.concatenate(all_knowledge, axis=0)  # [batch_size, knowledge_n]
    all_user_ids = np.array(all_user_ids)  # [batch_size]

    # Rebuild matrix aligned to actual user_id
    n_user = args.student_n
    n_know = args.knowledge_n
    stu_kn_pro_aligned = np.full((n_know, n_user), np.nan)  # [knowledge_n, n_user]

    for i, user_id in enumerate(all_user_ids):
        if user_id < n_user:
            stu_kn_pro_aligned[:, user_id] = stu_knowledge_matrix[i, :]

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    auc = roc_auc_score(all_labels, all_preds)

    # PR using original stu_knowledge_matrix
    pr_005 = cal_pr_value_05(stu_knowledge_matrix, adj_matrix)
    pr_01 = cal_pr_value_01(stu_knowledge_matrix, adj_matrix)

    # KRC using aligned matrix
    krc = calculate_krc(stu_exe, test_df, stu_kn_pro_aligned, Q_matrix, prob_desc)

    # Output results
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"ACC: {acc:.6f}")
    print(f"F1 : {f1:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"PR_0.05: {pr_005:.6f}")
    print(f"PR_0.1 : {pr_01:.6f}")
    print(f"KRC: {krc:.6f}")
    print("=" * 60)

    os.makedirs('result', exist_ok=True)
    with open('result/model_test.txt', 'a', encoding='utf8') as f:
        f.write(f'ACC= {acc:.6f}, F1= {f1:.6f}, RMSE= {rmse:.6f}, AUC= {auc:.6f}\n')
        f.write(f'PR_0.05= {pr_005:.6f}, PR_0.1= {pr_01:.6f}, KRC= {krc:.6f}\n')


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--student_n', type=int, default=672)
    params.add_argument('--exer_n', type=int, default=26)
    params.add_argument('--knowledge_n', type=int, default=16)
    params.add_argument('--d_edge_n', type=int, default=d_edge_n)
    params.add_argument('--node_features', type=int, default=8)
    params.add_argument('--edge_features', type=int, default=8)
    params.add_argument('--num_hidden', type=int, default=16)
    params.add_argument('--num_convs', type=int, default=1)
    args = params.parse_args()
    tes(args)