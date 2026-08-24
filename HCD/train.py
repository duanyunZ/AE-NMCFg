import gc
import os
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import sys

sys.path.append(os.getcwd())

from tools import Logger, labelize, to_numpy, divide_train_test
from config import DATA_PATH, hparams
from model import HierCDF


def load_prob_desc(data_path, dataset_name):
    """Load problemdesc.txt, return 'Obj'/'Sub' list"""
    prob_path = os.path.join(data_path, dataset_name, 'problemdesc.txt')
    if os.path.exists(prob_path):
        with open(prob_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        prob_desc = []
        for line in lines:
            if line in ['0', '1']:
                prob_desc.append('Obj' if line == '0' else 'Sub')
            else:
                prob_desc.append(line)
        return prob_desc
    else:
        return None


def build_stu_exe_from_data(data: pd.DataFrame, n_item: int, n_user: int):
    """Build response matrix (n_item, n_user) from long-format data.csv"""
    stu_exe = np.full((n_item, n_user), np.nan)
    for _, row in data.iterrows():
        user_id = int(row['user_id'])
        item_id = int(row['exer_id'])
        stu_exe[item_id, user_id] = row['score']
    return stu_exe


def train_test(data: pd.DataFrame, know_graph: pd.DataFrame = None,
               Q_matrix: np.array = None, stu_exe: np.array = None,
               prob_desc: list = None) -> dict:
    n_data = data.shape[0]
    n_user = hparams['n_user']
    n_item = hparams['n_item']
    n_know = hparams['n_know']
    hidden_dim = hparams['hidden_dim']
    device = hparams['device']
    logger_mode = hparams['logger_mode']
    train_ratio = hparams['train_ratio']
    log_path = hparams['log_path']
    itf_type = hparams['itf_type']
    model_name = hparams['model_name']

    # 8:2 split training and test sets
    train_data, test_data = divide_train_test(data, train_ratio)
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    model = HierCDF(
        n_user, n_item, n_know, hidden_dim, know_graph,
        itf_type, log_path, stu_exe, prob_desc
    )

    model.logger.write("{} {}".format(train_data.shape, test_data.shape), logger_mode)

    # Pass test set as valid_data (used for final evaluation)
    best_metrics = model.train(
        hparams=hparams,
        train_data=train_data,
        valid_data=test_data,
        Q_matrix=Q_matrix
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return best_metrics


if __name__ == '__main__':
    all_best_metrics = []

    # Load data
    data = pd.read_csv(osp.join(DATA_PATH, 'data.csv'))
    know_graph = pd.read_csv(osp.join(DATA_PATH, 'hier.csv'))
    Q_matrix = np.loadtxt(osp.join(DATA_PATH, 'Q_matrix.txt'), delimiter=' ')

    n_user = hparams['n_user']
    n_item = hparams['n_item']

    # Build full response matrix
    stu_exe = build_stu_exe_from_data(data, n_item, n_user)

    # Load prob_desc
    prob_desc = load_prob_desc(DATA_PATH, '')
    if prob_desc is None:
        # If problemdesc.txt does not exist, all exercises default to objective
        prob_desc = ['Obj'] * n_item
        print("Warning: problemdesc.txt not found, all exercises set to 'Obj'")

    # 10 repeated experiments
    for i in range(10):
        print(f"\n===== Run {i+1}/10 =====")
        best_metrics = train_test(data, know_graph, Q_matrix, stu_exe, prob_desc)

        # Print results
        print(f"Run {i+1}: ACC={best_metrics['acc']:.6f}, RMSE={best_metrics['rmse']:.6f}, "
              f"F1={best_metrics['f1']:.6f}, AUC={best_metrics['auc']:.6f}, "
              f"PR_0.05={best_metrics['pr_0.05']:.6f}, PR_0.1={best_metrics['pr_0.1']:.6f}, "
              f"KRC={best_metrics['krc']:.6f}")

        all_best_metrics.append(best_metrics)

    # Output summary statistics
    print("\n" + "=" * 80)
    print("Summary across 10 runs:")
    print("=" * 80)
    avg_acc = np.mean([m['acc'] for m in all_best_metrics])
    std_acc = np.std([m['acc'] for m in all_best_metrics])
    avg_rmse = np.mean([m['rmse'] for m in all_best_metrics])
    std_rmse = np.std([m['rmse'] for m in all_best_metrics])
    avg_krc = np.mean([m['krc'] for m in all_best_metrics if not np.isnan(m['krc'])])
    std_krc = np.std([m['krc'] for m in all_best_metrics if not np.isnan(m['krc'])])
    avg_pr05 = np.mean([m['pr_0.05'] for m in all_best_metrics])
    std_pr05 = np.std([m['pr_0.05'] for m in all_best_metrics])
    avg_pr01 = np.mean([m['pr_0.1'] for m in all_best_metrics])
    std_pr01 = np.std([m['pr_0.1'] for m in all_best_metrics])

    print(f"ACC: {avg_acc:.6f} ± {std_acc:.6f}")
    print(f"RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
    print(f"KRC: {avg_krc:.6f} ± {std_krc:.6f}")
    print(f"PR_0.05: {avg_pr05:.6f} ± {std_pr05:.6f}")
    print(f"PR_0.1 : {avg_pr01:.6f} ± {std_pr01:.6f}")