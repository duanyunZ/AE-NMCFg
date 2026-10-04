# -*- coding: utf-8 -*-
"""
Lipschitz-based optimization for AE-NMCFg.
"""

import os
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from scipy.sparse import coo_matrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)


def preprocess_adj(adj):
    """Convert adjacency matrix to PyG edge index. adj[i,j]=1 means i->j."""
    coo_adj = coo_matrix(adj)
    return torch.LongTensor(np.vstack([coo_adj.row, coo_adj.col]))


def get_structure_features(adj):
    """Construct H^(0): degree-aware features (κ=0.1) + bias padding (p=0.05)."""
    N = adj.shape[0]
    in_degree = np.sum(adj, axis=0, keepdims=True).T
    out_degree = np.sum(adj, axis=1, keepdims=True)

    D_max = max(np.max(in_degree), np.max(out_degree)) + 1e-8
    in_degree_norm = in_degree / D_max if np.max(in_degree) > 0 else in_degree
    out_degree_norm = out_degree / D_max if np.max(out_degree) > 0 else out_degree

    in_degree_sq = in_degree_norm ** 2
    out_degree_sq = out_degree_norm ** 2
    in_degree_log = np.log1p(in_degree)
    out_degree_log = np.log1p(out_degree)

    base_struct_feat = np.concatenate([
        in_degree, out_degree,
        in_degree_norm, out_degree_norm,
        in_degree_sq, out_degree_sq,
        in_degree_log, out_degree_log
    ], axis=1)

    struct_feat = base_struct_feat * 0.1
    bias_feat = np.ones((N, 2)) * 0.05
    node_feat = np.concatenate([struct_feat, bias_feat], axis=1)

    return torch.FloatTensor(node_feat)


class GAT_Knowledge(torch.nn.Module):
    """Two-layer GAT (10->8->10), no self-loops, for prerequisite strength learning."""
    def __init__(self, in_channels=10, hidden_channels=8, out_channels=10, heads=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                            dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        out, attn2 = self.gat2(x1, edge_index, return_attention_weights=True)
        return out, attn2


def get_R_from_attention(edge_index_attn, alpha, adj, kn_num, device):
    """Extract prerequisite strength matrix R from GAT attention (column-normalized)."""
    alpha_avg = alpha.cpu().detach().numpy()
    if alpha_avg.ndim > 1:
        alpha_avg = alpha_avg.mean(axis=1)

    R = np.zeros((kn_num, kn_num), dtype=np.float32)
    edges = edge_index_attn.cpu().numpy()

    for idx in range(edges.shape[1]):
        i, j = edges[0, idx], edges[1, idx]
        if adj[i, j] == 1:
            R[i, j] = alpha_avg[idx]

    for j in range(kn_num):
        s = np.sum(R[:, j])
        if s > 1e-8:
            R[:, j] /= s

    return torch.tensor(R, device=device, dtype=torch.float32)


def compute_cognitive_constraint_from_alpha(edge_index_attn, alpha, V, U, adj, device):
    """Eq. (7): constraint = Σ(Σ r·A_parent - A_child)."""
    kn_num = V.shape[1]
    M_st = U.shape[1]
    alpha_avg = alpha if alpha.dim() == 1 else alpha.mean(dim=1)

    sum_parent = torch.zeros((kn_num, M_st), device=device)
    valid_children = set()

    for idx in range(edge_index_attn.shape[1]):
        parent = edge_index_attn[0, idx].item()
        child = edge_index_attn[1, idx].item()
        if adj[parent, child] == 1:
            r = alpha_avg[idx]
            sum_parent[child] += r * torch.matmul(V[:, parent].unsqueeze(0), U).squeeze(0)
            valid_children.add(child)

    if not valid_children:
        return torch.tensor(0.0, device=device)

    children = list(valid_children)
    child_exp = torch.matmul(V[:, children].T, U)
    return torch.sum(sum_parent[children] - child_exp)


def compute_r_weighted_sum_from_alpha(edge_index_attn, alpha, V, U, adj, device):
    """Compute Σ r·Σ_m A_parent for GAT loss."""
    alpha_avg = alpha if alpha.dim() == 1 else alpha.mean(dim=1)
    total = torch.tensor(0.0, device=device)

    for idx in range(edge_index_attn.shape[1]):
        parent = edge_index_attn[0, idx].item()
        child = edge_index_attn[1, idx].item()
        if adj[parent, child] == 1:
            r = alpha_avg[idx]
            total += r * torch.sum(torch.matmul(V[:, parent].unsqueeze(0), U))

    return total


def compute_gat_reconstruction_loss_torch(h0, h_gat):
    """Eq. (8): R_gat = ||H^(2) - H^(0)||_F^2."""
    if h_gat.shape[1] >= h0.shape[1]:
        diff = h_gat[:, :h0.shape[1]] - h0
    else:
        h0_padded = torch.zeros_like(h_gat)
        h0_padded[:, :h0.shape[1]] = h0
        diff = h_gat - h0_padded
    return torch.sum(diff ** 2)


def cal_grad_U_m_torch(U, V, E, M, q_m, weight, train_data, train_fill, m,
                       lambda_reg, edge_index_attn, alpha, adj, device):
    """Gradient w.r.t. U[:, m] (Appendix Section 6)."""
    rank = U.shape[0]
    grad = torch.zeros(rank, device=device, dtype=torch.float32)

    q_m_t = torch.tensor(q_m, device=device, dtype=torch.float32)
    w_t = torch.tensor(weight, device=device, dtype=torch.float32)
    fill_t = torch.tensor(train_fill, device=device, dtype=torch.float32)
    data_t = torch.tensor(train_data, device=device, dtype=torch.float32)

    U_m = U[:, m]
    M_n = M.squeeze()

    # Negative log-likelihood gradient
    for n in range(train_data.shape[0]):
        if not np.isnan(train_data[n, m]):
            delta = torch.dot(q_m_t[n, :], torch.matmul(V.T, U_m)) + M_n[n]
            delta = torch.clamp(delta, -10, 10)
            phi = 0.5 * (1 + torch.erf(delta / math.sqrt(2)))
            phi = torch.clamp(phi, 0.001, 0.999)
            D = torch.exp(-delta**2 / 2) / (math.sqrt(2 * math.pi) * phi * (1 - phi) + 1e-6)
            D = torch.clamp(D, max=10.0)
            grad += D * (data_t[n, m] - phi) * torch.matmul(V, q_m_t[n, :].unsqueeze(1)).squeeze()
    grad = -grad

    # Reconstruction loss gradient: ||W⊙(X - EU)||_F^2
    grad += 2 * torch.matmul(E.T, w_t[:, m] * torch.matmul(E, U_m) - w_t[:, m] * fill_t[:, m])

    # HCA regularization gradient
    kn_num = V.shape[1]
    alpha_avg = alpha if alpha.dim() == 1 else alpha.mean(dim=1)
    sum_parent_v = torch.zeros((kn_num, rank), device=device)
    valid_children = set()

    for idx in range(edge_index_attn.shape[1]):
        parent = edge_index_attn[0, idx].item()
        child = edge_index_attn[1, idx].item()
        if adj[parent, child] == 1:
            r = alpha_avg[idx]
            sum_parent_v[child] += r * V[:, parent]
            valid_children.add(child)

    if valid_children:
        children = list(valid_children)
        grad += lambda_reg * torch.sum(V[:, children], dim=1)
        grad -= lambda_reg * torch.sum(sum_parent_v[children].T, dim=1)

    return grad


def cal_grad_V_torch(V, U, E, M, q_m, train_data, lambda_reg,
                     edge_index_attn, alpha, adj, device):
    """Gradient w.r.t. V (Appendix Section 6)."""
    ex_num, st_num = train_data.shape
    rank, kn_num = V.shape
    grad = torch.zeros_like(V, device=device)

    q_m_t = torch.tensor(q_m, device=device, dtype=torch.float32)
    data_t = torch.tensor(train_data, device=device, dtype=torch.float32)
    M_n = M.squeeze()

    # Negative log-likelihood gradient
    for n in range(ex_num):
        for m in range(st_num):
            if not np.isnan(train_data[n, m]):
                delta = torch.dot(q_m_t[n, :], torch.matmul(V.T, U[:, m])) + M_n[n]
                delta = torch.clamp(delta, -10, 10)
                phi = 0.5 * (1 + torch.erf(delta / math.sqrt(2)))
                phi = torch.clamp(phi, 0.001, 0.999)
                D = torch.exp(-delta**2 / 2) / (math.sqrt(2 * math.pi) * phi * (1 - phi) + 1e-6)
                D = torch.clamp(D, max=10.0)
                grad += D * (data_t[n, m] - phi) * torch.outer(U[:, m], q_m_t[n, :])
    grad = -grad

    # Reconstruction loss gradient: ||Q - EV||_F^2
    grad += 2 * torch.matmul(E.T, torch.matmul(E, V) - q_m_t)

    # HCA regularization gradient
    alpha_avg = alpha if alpha.dim() == 1 else alpha.mean(dim=1)
    edge_records, valid_children = [], set()

    for idx in range(edge_index_attn.shape[1]):
        parent = edge_index_attn[0, idx].item()
        child = edge_index_attn[1, idx].item()
        if adj[parent, child] == 1:
            r = alpha_avg[idx]
            edge_records.append((child, parent, r))
            valid_children.add(child)

    for m in range(st_num):
        Um = U[:, m]
        for c in valid_children:
            grad[:, c] += lambda_reg * Um
        for c, p, r in edge_records:
            grad[:, p] -= lambda_reg * r * Um

    return grad


def cal_grad_E_torch(E, U, V, weight, train_fill, q_m, device):
    """Gradient w.r.t. E: 2[W⊙(EU-X)]U^T + 2[EV-Q]V^T."""
    w_t = torch.tensor(weight, device=device, dtype=torch.float32)
    fill_t = torch.tensor(train_fill, device=device, dtype=torch.float32)
    q_t = torch.tensor(q_m, device=device, dtype=torch.float32)

    EU, EV = torch.matmul(E, U), torch.matmul(E, V)
    return 2 * torch.matmul(w_t * (EU - fill_t), U.T) + 2 * torch.matmul(EV - q_t, V.T)


def cal_grad_M_n_torch(n, U, V, M, q_m, train_data, device):
    """Gradient w.r.t. mu_n: -Σ D·(X - Φ(Δ))."""
    grad = torch.tensor(0.0, device=device, dtype=torch.float32)
    q_t = torch.tensor(q_m, device=device, dtype=torch.float32)
    data_t = torch.tensor(train_data, device=device, dtype=torch.float32)
    M_n = M[n, 0]

    for m in range(train_data.shape[1]):
        if not np.isnan(train_data[n, m]):
            delta = torch.dot(q_t[n, :], torch.matmul(V.T, U[:, m])) + M_n
            delta = torch.clamp(delta, -10, 10)
            phi = 0.5 * (1 + torch.erf(delta / math.sqrt(2)))
            phi = torch.clamp(phi, 0.001, 0.999)
            D = torch.exp(-delta**2 / 2) / (math.sqrt(2 * math.pi) * phi * (1 - phi) + 1e-6)
            D = torch.clamp(D, max=10.0)
            grad += D * (data_t[n, m] - phi)

    return -grad


def cal_lipschitz_constant(train_data, q_m, weight, U, E, V, M, device):
    """Lipschitz constants per Table I (computed once at initialization)."""
    ex_num, st_num = train_data.shape
    kn_num = V.shape[1]
    Lp = 1.0

    U_np, V_np, E_np = U.detach().cpu().numpy(), V.detach().cpu().numpy(), E.detach().cpu().numpy()

    sigma_U, sigma_W, sigma_V = np.linalg.norm(U_np, 2), np.linalg.norm(weight, 2), np.linalg.norm(V_np, 2)
    sigma_E = np.linalg.norm(E_np, 2)

    # L_U for each student
    l_u = np.zeros((st_num, 1), dtype=float)
    for m in range(st_num):
        C_m = V_np.dot(q_m.T)
        obs = [n for n in range(ex_num) if not np.isnan(train_data[n, m])]
        sigma_C = np.linalg.norm(C_m[:, obs], 2) if obs else 1.0
        l_u[m][0] = Lp * sigma_C**2 + 2 * sigma_E**2 * np.sqrt(np.sum(weight[:, m]**2)) + 1e-8

    # L_E
    l_e = 2 * sigma_U**2 * sigma_W + 2 * sigma_V**2

    # L_V for each knowledge concept
    l_v = np.zeros((kn_num, 1), dtype=float)
    for k in range(kn_num):
        C = 0.0
        for n in range(ex_num):
            for m in range(st_num):
                if not np.isnan(train_data[n, m]):
                    C += q_m[n, k]**2 * np.linalg.norm(U_np[:, m])**2
        l_v[k][0] = Lp * C + 2 * sigma_E**2 + 1e-8

    # L_mu for each exercise
    l_m = np.zeros((ex_num, 1), dtype=float)
    for n in range(ex_num):
        l_m[n][0] = Lp * sum(1 for m in range(st_num) if not np.isnan(train_data[n, m])) + 1e-8

    return l_u, l_e, l_v, l_m


def update_U_proj(U, grad_U, l_u, step_factor=1.0):
    """Projected gradient update for U."""
    with torch.no_grad():
        for m in range(U.shape[1]):
            step = step_factor / l_u[m][0]
            U[:, m] = U[:, m] - step * grad_U[:, m]
            U[:, m].clamp_(min=0)
    return U


def update_V_proj(V, grad_V, l_v, step_factor=1.0):
    """Projected gradient update for V."""
    with torch.no_grad():
        for k in range(V.shape[1]):
            step = step_factor / l_v[k][0]
            V[:, k] = V[:, k] - step * grad_V[:, k]
            V[:, k].clamp_(min=0)
    return V


def update_E_proj(E, grad_E, l_e, step_factor=1.0):
    """Projected gradient update for E."""
    with torch.no_grad():
        step = step_factor / l_e
        E = E - step * grad_E
        E.clamp_(min=0)
    return E


def update_M_proj(M, grad_M, l_m, step_factor=1.0):
    """Gradient descent update for M (unconstrained)."""
    with torch.no_grad():
        for n in range(M.shape[0]):
            step = step_factor / l_m[n][0]
            M[n, 0] = M[n, 0] - step * grad_M[n]
    return M


def compute_likelihood_loss_torch(weight, train_fill, Q, U, V, M, device):
    """Negative log-likelihood under probit model."""
    w_t = torch.tensor(weight, device=device, dtype=torch.float32)
    fill_t = torch.tensor(train_fill, device=device, dtype=torch.float32)
    Q_t = torch.tensor(Q, device=device, dtype=torch.float32)

    delta = torch.matmul(torch.matmul(Q_t, V.T), U) + M
    delta = torch.clamp(delta, -20, 20)
    phi = 0.5 * (1 + torch.erf(delta / math.sqrt(2)))
    phi = torch.clamp(phi, 1e-5, 0.99999)

    return -torch.sum(w_t * fill_t * torch.log(phi) + w_t * (1 - fill_t) * torch.log(1 - phi))


def compute_reconstruction_loss_torch(U, V, E, train_fill, weight, Q, device):
    """Reconstruction loss: ||W⊙(X-EU)||_F^2 + ||Q-EV||_F^2."""
    w_t = torch.tensor(weight, device=device, dtype=torch.float32)
    fill_t = torch.tensor(train_fill, device=device, dtype=torch.float32)
    Q_t = torch.tensor(Q, device=device, dtype=torch.float32)

    EU, EV = torch.matmul(E, U), torch.matmul(E, V)
    return torch.norm(w_t * (fill_t - EU))**2 + torch.norm(Q_t - EV)**2


def fit_data(train_data, train_fill, q_m, weight, rank, adj_matrix,
             lambda_reg, beta_reg, max_iter, cri,
             early_stop_delta_threshold=1.0,
             early_stop_patience=6,
             early_stop_patience_no_improve=50,
             early_stop_enabled=True):
    """Main training loop: PG-BCD for U,V,E,M + Adam for GAT."""
    ex_num, st_num = train_data.shape
    kn_num = q_m.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize matrix factors
    U = torch.rand(rank, st_num, device=device, dtype=torch.float32) * 1.0
    E = torch.rand(ex_num, rank, device=device, dtype=torch.float32) * 1.0
    V = torch.rand(rank, kn_num, device=device, dtype=torch.float32) * 1.0
    M = torch.rand(ex_num, 1, device=device, dtype=torch.float32) * 1.0

    # Initialize GAT
    h0 = get_structure_features(adj_matrix).to(device)
    edge_index = preprocess_adj(adj_matrix).to(device)

    gat_model = GAT_Knowledge().to(device)
    gat_optimizer = optim.Adam(gat_model.parameters(), lr=0.0001, weight_decay=1e-5)

    gat_model.eval()
    with torch.no_grad():
        _, attn = gat_model(h0, edge_index)
        edge_index_attn, alpha = attn
        R = get_R_from_attention(edge_index_attn, alpha, adj_matrix, kn_num, device)

    # Lipschitz constants (computed once)
    l_u, l_e, l_v, l_m = cal_lipschitz_constant(train_data, q_m, weight, U, E, V, M, device)
    print(f"L_U=[{np.min(l_u):.2e}, {np.max(l_u):.2e}], L_E={l_e:.2e}, L_V=[{np.min(l_v):.2e}, {np.max(l_v):.2e}]")

    # Initial objective (Eq. 9)
    obj_old = (compute_likelihood_loss_torch(weight, train_fill, q_m, U, V, M, device)
               + compute_reconstruction_loss_torch(U, V, E, train_fill, weight, q_m, device)
               - lambda_reg * compute_cognitive_constraint_from_alpha(edge_index_attn, alpha, V, U, adj_matrix, device)
               + beta_reg * compute_gat_reconstruction_loss_torch(h0, h0))

    best_obj = obj_old.item()
    best_params = (U.detach().cpu().numpy(), E.detach().cpu().numpy(),
                   V.detach().cpu().numpy(), M.detach().cpu().numpy(), R.detach().cpu().numpy())
    best_delta, best_delta_params, best_delta_iter = float('inf'), best_params, 0
    no_improve_count, prev_delta, consecutive_dec = 0, None, 0
    step_factor = 1.0

    print(f"Initial objective = {obj_old:.6f}")

    for i in range(1, max_iter + 1):
        # Compute gradients
        grad_U = torch.zeros_like(U)
        for m in range(st_num):
            grad_U[:, m] = cal_grad_U_m_torch(U, V, E, M, q_m, weight, train_data, train_fill, m,
                                               lambda_reg, edge_index_attn, alpha, adj_matrix, device)

        grad_V = cal_grad_V_torch(V, U, E, M, q_m, train_data, lambda_reg,
                                   edge_index_attn, alpha, adj_matrix, device)
        grad_E = cal_grad_E_torch(E, U, V, weight, train_fill, q_m, device)
        grad_M = torch.zeros(ex_num, 1, device=device)
        for n in range(ex_num):
            grad_M[n, 0] = cal_grad_M_n_torch(n, U, V, M, q_m, train_data, device)

        U = update_U_proj(U, grad_U, l_u, step_factor)
        V = update_V_proj(V, grad_V, l_v, step_factor)
        E = update_E_proj(E, grad_E, l_e, step_factor)
        M = update_M_proj(M, grad_M, l_m, step_factor)

        gat_model.train()
        h_gat, attn = gat_model(h0, edge_index)
        edge_index_attn, alpha = attn
        gat_loss = (beta_reg * compute_gat_reconstruction_loss_torch(h0, h_gat)
                    - lambda_reg * compute_r_weighted_sum_from_alpha(edge_index_attn, alpha, V, U, adj_matrix, device))
        gat_optimizer.zero_grad()
        gat_loss.backward()
        torch.nn.utils.clip_grad_norm_(gat_model.parameters(), 1.0)
        gat_optimizer.step()

        gat_model.eval()
        with torch.no_grad():
            _, attn = gat_model(h0, edge_index)
            edge_index_attn, alpha = attn
            R_new = get_R_from_attention(edge_index_attn, alpha, adj_matrix, kn_num, device)

        constraint_full = compute_cognitive_constraint_from_alpha(edge_index_attn, alpha, V, U, adj_matrix, device)
        obj_new = (compute_likelihood_loss_torch(weight, train_fill, q_m, U, V, M, device)
                   + compute_reconstruction_loss_torch(U, V, E, train_fill, weight, q_m, device)
                   - lambda_reg * constraint_full
                   + beta_reg * compute_gat_reconstruction_loss_torch(h0, h_gat))

        if torch.isnan(obj_new) or torch.isinf(obj_new):
            print(f"Invalid objective at iter {i}, using best params")
            break

        delta = obj_old - obj_new
        obj_old = obj_new

        # Track best
        if obj_new < best_obj:
            best_obj = obj_new
            best_params = (U.detach().cpu().numpy(), E.detach().cpu().numpy(),
                           V.detach().cpu().numpy(), M.detach().cpu().numpy(),
                           R_new.detach().cpu().numpy())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if delta < best_delta:
            best_delta, best_delta_params, best_delta_iter = delta, best_params, i

        stop_flag, stop_reason = False, None
        if delta < 0:
            stop_flag, stop_reason = True, f"obj increased (delta={delta:.6f})"
        elif early_stop_enabled and prev_delta is not None:
            if abs(delta - prev_delta) < early_stop_delta_threshold:
                stop_flag, stop_reason = True, f"delta converged (|{delta:.6f}-{prev_delta:.6f}|<{early_stop_delta_threshold})"
            elif delta < prev_delta:
                consecutive_dec += 1
            elif consecutive_dec >= early_stop_patience:
                stop_flag, stop_reason = True, f"trend reversal (delta={delta:.6f}, prev={prev_delta:.6f})"
            else:
                consecutive_dec = max(0, consecutive_dec - 1)

        if early_stop_enabled and not stop_flag and no_improve_count >= early_stop_patience_no_improve:
            stop_flag, stop_reason = True, f"no improvement for {early_stop_patience_no_improve} iters"

        if stop_flag and early_stop_enabled:
            final_params = best_delta_params if "trend reversal" in stop_reason else best_params
            print(f"Early stop at iter {i}: {stop_reason}")
            return final_params

        prev_delta = delta
        print(f"Iter {i}: obj={obj_new:.6f}, delta={delta:.6f}")

    return best_params
