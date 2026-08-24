import sys
import os
import gc
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import time
import torch
import torch.nn as nn
import warnings
from scipy.stats import wilcoxon

from config import hparams, DATA_PATH
from dataloader import TrainDataLoader
from itf import mirt2pl, sigmoid_dot, dot, itf_dict
from tools import Logger, df_preview, labelize, to_numpy

# Import KRC calculation functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


class HierCDF(nn.Module):
    '''
    The hierarchical cognitive diagnosis model
    '''

    def __init__(self, n_user, n_item, n_know, hidden_dim, know_graph: pd.DataFrame, itf_type='mirt', \
                 log_path='./log/', stu_exe=None, prob_desc=None):

        super(HierCDF, self).__init__()
        self.logger = Logger(path=log_path)

        self.n_user = n_user
        self.n_item = n_item
        self.n_know = n_know
        self.hidden_dim = hidden_dim
        self.know_graph = know_graph

        self.stu_exe = stu_exe
        self.prob_desc = prob_desc

        # Build directed graph
        self.know_edge = nx.DiGraph()
        for k in range(n_know):
            self.know_edge.add_node(k)

        self.parent_child_pairs = []
        for edge in know_graph.values.tolist():
            parent, child = edge[0], edge[1]
            if parent < n_know and child < n_know:
                self.know_edge.add_edge(parent, child)
                self.parent_child_pairs.append((parent, child))
            else:
                self.logger.write(f"Warning: Invalid edge ({parent}, {child}) exceeds n_know={n_know}", 'console')

        try:
            self.topo_order = list(nx.topological_sort(self.know_edge))
        except nx.NetworkXError:
            self.logger.write("Error: Cycle detected in knowledge graph!", 'console')
            self.topo_order = list(range(n_know))

        # Conditional mastery parameters
        condi_p = torch.Tensor(n_user, n_know)
        self.condi_p = nn.Parameter(condi_p)

        condi_n = torch.Tensor(n_user, n_know)
        self.condi_n = nn.Parameter(condi_n)

        priori = torch.Tensor(n_user, n_know)
        self.priori = nn.Parameter(priori)

        # item representation
        self.item_diff = nn.Embedding(n_item, n_know)
        self.item_disc = nn.Embedding(n_item, 1)

        # embedding transformation
        self.user_contract = nn.Linear(n_know, hidden_dim)
        self.item_contract = nn.Linear(n_know, hidden_dim)

        # Neural Interaction Module (used only in ncd)
        self.cross_layer1 = nn.Linear(hidden_dim, max(int(hidden_dim / 2), 1))
        self.cross_layer2 = nn.Linear(max(int(hidden_dim / 2), 1), 1)

        # Directly set itf without calling set_itf method
        self.itf_type = itf_type
        self.itf = itf_dict.get(itf_type, self.ncd)

        # param initialization
        nn.init.xavier_normal_(self.priori)
        nn.init.xavier_normal_(self.condi_p)
        nn.init.xavier_normal_(self.condi_n)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def ncd(self, user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
        input_vec = (user_emb - item_emb) * item_offset
        x_vec = torch.sigmoid(self.cross_layer1(input_vec))
        x_vec = torch.sigmoid(self.cross_layer2(x_vec))
        return x_vec

    def set_itf(self, itf_type):
        self.itf_type = itf_type
        self.itf = itf_dict.get(itf_type, self.ncd)

    def get_posterior(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        posterior = torch.zeros(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_p = torch.sigmoid(self.condi_p[user_ids, :])
        batch_condi_n = torch.sigmoid(self.condi_n[user_ids, :])

        for k in self.topo_order:
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)

            if len_p == 0:
                priori = batch_priori[:, k]
                posterior[:, k] = priori.reshape(-1)
                continue

            fmt = '{0:0%db}' % (len_p)
            n_condi = 2 ** len_p

            priori = posterior[:, predecessors]
            condi_p = torch.pow(batch_condi_p[:, :len_p], 1 / len_p) if len_p > 0 else batch_condi_p
            condi_n = torch.pow(batch_condi_n[:, :len_p], 1 / len_p) if len_p > 0 else batch_condi_n

            margin_p = condi_p * priori
            margin_n = condi_n * (1.0 - priori)

            posterior_k = torch.zeros((1, n_batch)).to(device)

            for idx in range(n_condi):
                mask = fmt.format(idx)
                mask = torch.Tensor(np.array(list(mask)).astype(int)).to(device)

                margin = mask * margin_p + (1 - mask) * margin_n
                margin = torch.prod(margin, dim=1).unsqueeze(dim=0)

                posterior_k = torch.cat([posterior_k, margin], dim=0)
            posterior_k = (torch.sum(posterior_k, dim=0)).squeeze()

            posterior[:, k] = posterior_k.reshape(-1)

        return posterior

    def get_condi_p(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.zeros(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_p = torch.sigmoid(self.condi_p[user_ids, :])

        for k in self.topo_order:
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            condi_p = torch.pow(batch_condi_p[:, :len_p], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_p, dim=1).reshape(-1)

        return result_tensor

    def get_condi_n(self, user_ids: torch.LongTensor, device='cpu') -> torch.Tensor:
        n_batch = user_ids.shape[0]
        result_tensor = torch.zeros(n_batch, self.n_know).to(device)
        batch_priori = torch.sigmoid(self.priori[user_ids, :])
        batch_condi_n = torch.sigmoid(self.condi_n[user_ids, :])

        for k in self.topo_order:
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            condi_n = torch.pow(batch_condi_n[:, :len_p], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_n, dim=1).reshape(-1)

        return result_tensor

    def concat(self, a, b, dim=0):
        if a is None:
            return b.reshape(-1, 1)
        else:
            return torch.cat([a, b], dim=dim)

    def forward(self, user_ids: torch.LongTensor, item_ids: torch.LongTensor, item_know: torch.Tensor,
                device='cpu') -> torch.Tensor:
        user_mastery = self.get_posterior(user_ids, device)
        item_diff = torch.sigmoid(self.item_diff(item_ids))
        item_disc = torch.sigmoid(self.item_disc(item_ids))

        user_factor = torch.tanh(self.user_contract(user_mastery * item_know))
        item_factor = torch.sigmoid(self.item_contract(item_diff * item_know))

        output = self.itf(user_factor, item_factor, item_disc)

        return output

    def calculate_krc(self, test_data: pd.DataFrame, Q_matrix: np.array, device='cpu') -> float:
        if self.stu_exe is None or self.prob_desc is None:
            self.logger.write("Warning: stu_exe or prob_desc is None, KRC cannot be computed.", 'console')
            return np.nan

        with torch.no_grad():
            all_user_ids = torch.arange(self.n_user).to(device)
            stu_kn_pro = self.get_posterior(all_user_ids, device)
            stu_kn_pro = stu_kn_pro.detach().cpu().numpy().T

        miss_coo = []
        for idx in range(len(test_data)):
            user_id = int(test_data.iloc[idx]['user_id'])
            item_id = int(test_data.iloc[idx]['exer_id'])
            miss_coo.append((item_id, user_id))

        kn_krc_list = ev.cal_diag_krc(
            self.prob_desc,
            self.stu_exe,
            miss_coo,
            stu_kn_pro,
            Q_matrix
        )

        valid_krc = [v for v in kn_krc_list.values() if not np.isnan(v)]
        if len(valid_krc) > 0:
            return np.mean(valid_krc)
        else:
            return np.nan

    def calculate_pr(self, alpha: float = 0.05, device='cpu') -> float:
        if len(self.parent_child_pairs) == 0:
            return 0.0

        with torch.no_grad():
            all_user_ids = torch.arange(self.n_user).to(device)
            posterior = self.get_posterior(all_user_ids, device)
            posterior = posterior.detach().cpu().numpy()

        p_values = []
        for i in range(self.n_user):
            student_mastery = posterior[i]
            differences = []
            for parent, child in self.parent_child_pairs:
                if parent < self.n_know and child < self.n_know:
                    diff = student_mastery[parent] - student_mastery[child]
                    differences.append(diff)

            if len(differences) == 0:
                p_values.append(1.0)
                continue

            differences = np.array(differences)

            if np.all(np.abs(differences) < 1e-10):
                p_value = 1.0
            elif len(differences) == 1:
                p_value = 0.5 if differences[0] > 0 else 1.0
            else:
                try:
                    _, p_value = wilcoxon(differences, alternative='greater')
                except Exception as e:
                    self.logger.write(f"Wilcoxon test failed for student {i}: {e}", 'console')
                    p_value = 1.0

            p_values.append(p_value)

        pr = np.mean(np.array(p_values) < alpha)
        return pr

    def train(self, hparams: dict, train_data: pd.DataFrame, Q_matrix: np.array,
              valid_data: pd.DataFrame = None):
        lr = hparams.get('lr', 0.01)
        epoch = hparams.get('epoch', 5)
        batch_size = hparams.get('batch_size', 64)
        logger_mode = hparams.get('logger_mode', 'both')
        loss_factor = hparams.get('loss_factor', 1.0)
        device = hparams.get('device', 'cpu')
        batch_show = hparams.get('batch_show', 200)

        self.logger.write('Before train. hparams = {}'.format(str(hparams)), logger_mode)

        self.to(device)

        loss_fn = MyLoss(self, nn.NLLLoss, loss_factor)

        dataloader = TrainDataLoader(train_data, Q_matrix, batch_size)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        y_target_all = np.array(train_data.loc[:, 'score']).astype(int)

        best_metrics = {
            'epoch': -1,
            'acc': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'rmse': float('inf'),
            'pr_0.05': 0.0,
            'pr_0.1': 0.0,
            'krc': np.nan
        }

        for step in range(1, epoch + 1):
            dataloader.reset()
            loss_all = 0.0
            batch_count = 0
            y_pred_all = np.array([])

            while not dataloader.is_end():
                batch_count += 1
                optimizer.zero_grad()
                user_ids, item_ids, item_know, y_target = dataloader.next_batch()
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                item_know = item_know.to(device)
                y_target = y_target.to(device)

                y_pred = self.forward(user_ids, item_ids, item_know, device)

                output_1 = y_pred
                output_0 = torch.ones(output_1.size()).to(device) - output_1
                output = torch.cat((output_0, output_1), 1)
                loss = loss_fn(torch.log(output), y_target, user_ids)
                loss.backward()
                optimizer.step()

                self.pos_clipper([self.user_contract, self.item_contract])
                self.pos_clipper([self.cross_layer1, self.cross_layer2])

                y_pred_batch = labelize(y_pred)
                y_pred_all = np.concatenate([y_pred_all, y_pred_batch], axis=0)

                loss_all += loss.item()

                if batch_count % batch_show == batch_show - 1:
                    self.logger.write('epoch = {}, batch = {}, loss = {}'.format(
                        step, batch_count, loss_all / batch_show), logger_mode)
                    loss_all = 0.0

            train_f1 = f1_score(y_target_all, y_pred_all)
            self.logger.write('epoch = {}, train_f1 = {}'.format(step, train_f1), logger_mode)

            if valid_data is not None:
                metrics = self.validate_with_krc(
                    valid_data, Q_matrix, device, logger_mode,
                    epoch=step, total_epoch=epoch
                )

                if metrics['acc'] > best_metrics['acc']:
                    best_metrics['epoch'] = step
                    best_metrics['acc'] = metrics['acc']
                    best_metrics['f1'] = metrics['f1']
                    best_metrics['auc'] = metrics['auc']
                    best_metrics['rmse'] = metrics['rmse']
                    if 'pr_0.05' in metrics:
                        best_metrics['pr_0.05'] = metrics['pr_0.05']
                    if 'pr_0.1' in metrics:
                        best_metrics['pr_0.1'] = metrics['pr_0.1']
                    if 'krc' in metrics and not np.isnan(metrics['krc']):
                        best_metrics['krc'] = metrics['krc']

        self.logger.write('=' * 60, logger_mode)
        self.logger.write('Best Performance Metrics:', logger_mode)
        self.logger.write('=' * 60, logger_mode)
        self.logger.write(f'Best epoch: {best_metrics["epoch"]}', logger_mode)
        self.logger.write(f'Best ACC: {best_metrics["acc"]:.6f}', logger_mode)
        self.logger.write(f'Best F1 : {best_metrics["f1"]:.6f}', logger_mode)
        self.logger.write(f'Best AUC: {best_metrics["auc"]:.6f}', logger_mode)
        self.logger.write(f'Best RMSE: {best_metrics["rmse"]:.6f}', logger_mode)
        if best_metrics['pr_0.05'] > 0:
            self.logger.write(f'PR_0.05: {best_metrics["pr_0.05"]:.6f}', logger_mode)
        if best_metrics['pr_0.1'] > 0:
            self.logger.write(f'PR_0.1 : {best_metrics["pr_0.1"]:.6f}', logger_mode)
        if not np.isnan(best_metrics['krc']):
            self.logger.write(f'KRC: {best_metrics["krc"]:.6f}', logger_mode)
        self.logger.write('=' * 60, logger_mode)

        return best_metrics

    def pos_clipper(self, module_list: list):
        for module in module_list:
            module.weight.data = module.weight.clamp_min(0)
        return

    def predict(self, data: pd.DataFrame, Q_matrix: np.array, device='cpu') -> pd.DataFrame:
        dataloader = TrainDataLoader(data, Q_matrix, 8192)
        dataloader.reset()
        df_pred = pd.DataFrame(columns=['predict_score', 'predict_label'])
        self.to(device)

        with torch.no_grad():
            while not dataloader.is_end():
                user_ids, item_ids, item_know, _ = dataloader.next_batch()
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                item_know = item_know.to(device)

                z_output = self.forward(user_ids, item_ids, item_know, device=device)
                z_score = to_numpy(z_output).reshape(-1)
                z_label = labelize(z_output).reshape(-1)
                df_batch = pd.DataFrame({
                    'predict_score': z_score,
                    'predict_label': z_label
                })
                df_pred = pd.concat([df_pred, df_batch], ignore_index=True)

        result = data.reset_index().join(df_pred)
        return result

    def validate_with_krc(self, valid_data: pd.DataFrame, Q_matrix: np.array, device, logger_mode,
                          epoch=None, total_epoch=None):
        valid_pred = self.predict(valid_data, Q_matrix, device)
        z_true = valid_pred['score'].astype(int).tolist()
        z_score = valid_pred['predict_score'].tolist()
        z_label = valid_pred['predict_label'].tolist()

        valid_acc = accuracy_score(z_true, z_label)
        valid_f1 = f1_score(z_true, z_label)
        valid_auc = roc_auc_score(z_true, z_score)
        valid_mse = mean_squared_error(z_true, z_score)
        valid_rmse = np.sqrt(valid_mse)

        epoch_info = f"epoch = {epoch}/{total_epoch}" if epoch is not None else ""
        self.logger.write(f'valid acc = {valid_acc:.6f} {epoch_info}', logger_mode)
        self.logger.write(f'valid f1  = {valid_f1:.6f} {epoch_info}', logger_mode)
        self.logger.write(f'valid auc = {valid_auc:.6f} {epoch_info}', logger_mode)
        self.logger.write(f'valid rmse = {valid_rmse:.6f} {epoch_info}', logger_mode)

        metrics_dict = {
            'acc': valid_acc,
            'f1': valid_f1,
            'auc': valid_auc,
            'mse': valid_mse,
            'rmse': valid_rmse
        }

        self.logger.write('Calculating PR and KRC...', logger_mode)

        pr_05 = self.calculate_pr(alpha=0.05, device=device)
        pr_01 = self.calculate_pr(alpha=0.1, device=device)
        self.logger.write(f'PR_0.05 = {pr_05:.6f}', logger_mode)
        self.logger.write(f'PR_0.1 = {pr_01:.6f}', logger_mode)
        metrics_dict['pr_0.05'] = pr_05
        metrics_dict['pr_0.1'] = pr_01

        krc = self.calculate_krc(valid_data, Q_matrix, device)
        if not np.isnan(krc):
            self.logger.write(f'KRC = {krc:.6f}', logger_mode)
            metrics_dict['krc'] = krc
        else:
            self.logger.write('KRC: NaN (stu_exe or prob_desc missing)', logger_mode)
            metrics_dict['krc'] = np.nan

        self.logger.write('', logger_mode)

        return metrics_dict

    def save(self, model_name='./model.pkl'):
        path = '/'.join(model_name.split('/')[:-1]) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), model_name)

    def load(self, model_name='./model.pkl'):
        self.load_state_dict(torch.load(model_name))


class MyLoss(nn.Module):
    def __init__(self, net: HierCDF, loss_fn: nn.Module, factor=1.0):
        super(MyLoss, self).__init__()
        self.net = net
        self.factor = factor
        self.loss_fn = loss_fn()

    def forward(self, y_pred, y_target, user_ids):
        return self.loss_fn(y_pred, y_target) + self.factor * torch.sum(
            torch.relu(self.net.condi_n[user_ids, :] - self.net.condi_p[user_ids, :]))


def test(hparams):
    n_user = hparams['n_user']
    n_item = hparams['n_item']
    n_know = hparams['n_know']
    hidden_dim = hparams['hidden_dim']

    data = pd.read_csv(DATA_PATH + 'data_demo.csv', index_col=0)
    data = data.sample(frac=1).reset_index(drop=True)
    know_graph = pd.read_csv(DATA_PATH + 'hierarchy_demo.csv', index_col=0)
    Q_matrix = np.loadtxt(DATA_PATH + 'Q_matrix_demo.txt', delimiter=' ')

    net = HierCDF(n_user, n_item, n_know, hidden_dim, know_graph)

    net.train(hparams=hparams, Q_matrix=Q_matrix, train_data=data)


if __name__ == '__main__':
    test(hparams)