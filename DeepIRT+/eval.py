from moduel.LSTM import LSTM
from data.loader import DataLoader
from model.DeepModel import DeepIRT
import tqdm
import numpy as np
from sklearn import metrics
import pickle
import torch
import torch.optim as optim
import torch.nn as nn


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, pred, gold):
        loss = 0
        gold = [1 if elem == 1 else 0 for elem in gold]
        for i in range(len(gold)):
            loss += gold[i] * torch.log(pred[i]) + (1 - gold[i]) * torch.log(1 - pred[i])
        return -loss


def performance(p, g):
    p = np.array(p)
    g = np.array(g)
    print(p, g)
    pred = [1 if elem >= 0.5 else 0 for elem in p]
    gold = [1 if elem == 1 else 0 for elem in g]
    res = [elem for elem in np.abs(np.array(pred) - np.array(gold))]

    # fpr, tpr, thresholds = metrics.roc_curve(g, p)
    # auc = metrics.auc(fpr, tpr)
    # f1 = metrics.f1_score(gold, pred)
    # recall = metrics.recall_score(gold, pred)
    # precision = metrics.precision_score(gold, pred)

    acc = 1 - sum(res) / len(res)
    mse = metrics.mean_squared_error(g, p)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(gold, pred)
    return acc, rmse, mae


def train_epoch(model, training_data, optimizer, loss_func):
    for batch in tqdm.tqdm(training_data, desc='Training:    ', mininterval=2):
        (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
        pred = model(uididx, kcodeidx, qidemb, qidemblength, 'train')
        loss = loss_func(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(data, model, validation_data, loss_func,
               uid_index_dict, kcode_index_dict, plm_intro, q_matrix):
    p = []  # prediction results (vector)
    g = []  # ground-of-truth results (vector)
    p_obj = []  # prediction results for objective exercises
    g_obj = []  # ground-of-truth results for objective exercises

    stu_kn_pro = None

    if data in ['Math1', 'Math2']:  # because these datasets include subjective exercises.

        # 1. rmse and mae
        for batch in tqdm.tqdm(validation_data, desc='Testing:    ', mininterval=2):
            (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
            pred = model(uididx, kcodeidx, qidemb, qidemblength, 'train')
            p += list(pred.view(len(pred)).data.numpy())
            g += list(Y.data.numpy())

            obj_list = []
            c = 0
            for elem in kcodeidx:
                elem = elem.tolist()
                # get the ID of the knowledge concepts in elem
                kn_ids = [int(get_keys(kcode_index_dict, _)) for _ in elem]
                kn_ids.sort()
                for ex in range(q_matrix.shape[0]):
                    ex_kns = np.where(q_matrix[ex] == 1)[0].tolist()
                    if ex_kns == kn_ids:
                        # find the exercise
                        if plm_intro[ex][0] == 0:  # '0' denotes the objective exercise
                            obj_list.append(c)
                        break
                c += 1
            p_obj += [p[_] for _ in range(len(p)) if _ in obj_list]
            g_obj += [g[_] for _ in range(len(g)) if _ in obj_list]

        acc, rmse, mae = performance(p, g)
        acc_obj, rmse_obj, mae_obj = performance(p_obj, g_obj)

        if data in ['FrcSub', 'Junyi-s', 'ednet', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
            # 2. the student-knowledge proficiency matrix based on true student and knowledeg IDs
            uididx_list = torch.tensor([_ for _ in range(len(uid_index_dict))])
            kcodeidx_list = torch.tensor([_ for _ in range(len(kcode_index_dict))])
            uididx_kcodeidx_pro = model.knowledge_proficiency(uididx_list, kcodeidx_list)
            stu_kn_pro = np.nan * np.ones([len(uid_index_dict), len(kcode_index_dict)])
            for uid_idx, uid_pros in uididx_kcodeidx_pro.items():
                for kcode_idx, mastery in uid_pros.items():
                    stu_kn_pro[uid_idx][kcode_idx] = mastery
            # ================================================================
    else:
        # 1. rmse and mae
        for batch in tqdm.tqdm(validation_data, desc='Testing:    ', mininterval=2):
            (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
            pred = model(uididx, kcodeidx, qidemb, qidemblength, 'train')
            p += list(pred.view(len(pred)).data.numpy())
            g += list(Y.data.numpy())

        acc_obj, rmse, mae = performance(p, g)

        if data in ['FrcSub', 'ednet', 'Junyi-s', 'assist09','Quanlang-s','unit-bio-small','unit-his-small','unit-eng']:
            # 2. the student-knowledge proficiency matrix based on true student and knowledeg IDs
            uididx_list = torch.tensor([_ for _ in range(len(uid_index_dict))])
            kcodeidx_list = torch.tensor([_ for _ in range(len(kcode_index_dict))])
            uididx_kcodeidx_pro = model.knowledge_proficiency(uididx_list, kcodeidx_list)
            stu_kn_pro = np.nan * np.ones([len(uid_index_dict), len(kcode_index_dict)])
            for uid_idx, uid_pros in uididx_kcodeidx_pro.items():
                for kcode_idx, mastery in uid_pros.items():
                    stu_kn_pro[uid_idx][kcode_idx] = mastery
            # ================================================================

    return acc_obj, rmse, mae, stu_kn_pro


def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k