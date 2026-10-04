# -*- coding: utf-8 -*-
"""
Toward Monotonicity and Hierarchical Rationality in Student Cognitive Modeling via Autoencoder-like Matrix Co-Factorization.

This model jointly factorizes the student response matrix X (N x M) and the Q-matrix (N x K)
into nonnegative latent factors E (N x T), U (T x M), and V (T x K). The knowledge proficiency
matrix A = V^T U is reconstructed via a probit link function to enforce psychometric monotonicity.
A Graph Attention Network (GAT) learns adaptive prerequisite strengths from the knowledge
prerequisite graph, and a hierarchical regularizer enforces the Hierarchical Cognitive Assumption
(HCA), i.e., mastery of prerequisite knowledge concepts bounds the mastery of descendants.

"""

import warnings
import ae_nmcfg_lipschitz


class ae_nmcfg():
    """
    AE-NMCFg Model.

    The model integrates:
        1. Matrix co-factorization: jointly factorizes X and Q into E, U, V.
        2. Probit reconstruction: maps A = V^T U to response probabilities via Phi(·).
        3. GAT-based prerequisite learner: learns adaptive prerequisite strengths R.
        4. Hierarchical regularization: enforces HCA via R_hir = Σ(A_j - Σ r_i->j A_i).
        5. Graph self-reconstruction loss: R_gat = ||H^(2) - H^(0)||_F^2.

    Attributes:
        train_data (np.ndarray): Response matrix with NaN for missing entries (N x M).
        train_fill (np.ndarray): Response matrix with missing entries filled (N x M).
        Q (np.ndarray): Q-matrix, binary exercise-knowledge associations (N x K).
        W (np.ndarray): Binary mask, 1 for observed entries, 0 for missing (N x M).
        rank (int): Number of latent topics T.
        adj_matrix (np.ndarray): Knowledge prerequisite graph adjacency (K x K),
                                  adj[i, j] = 1 means i -> j (i is prerequisite of j).
        lambda_reg (float): Weight for the hierarchical HCA regularizer.
        beta_reg (float): Weight for the GAT self-reconstruction loss.
    """

    def __init__(self, train_data, train_fill, q_m, weight, rank,
                 adj_matrix, lambda_reg=1.0, beta_reg=0.1):
        self.train_data = train_data
        self.train_fill = train_fill
        self.Q = q_m
        self.W = weight
        self.rank = rank
        self.adj_matrix = adj_matrix
        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg

    def train(self, max_iter=500, cri=1e-6,
              early_stop_delta_threshold=None,
              early_stop_patience=3,
              early_stop_patience_no_improve=50,
              early_stop_enabled=True):
        """
        Train the AE-NMCFg model.

        The optimization follows a block coordinate descent (BCD) scheme:
            - Matrix factors (U, E, V, M) are updated via projected gradient descent
              with Lipschitz-based step sizes (Theorem 1 and Table I).
            - GAT parameters (W_gat, alpha) are updated via Adam backpropagation.

        Args:
            max_iter (int): Maximum number of outer iterations. Default: 500.
            cri (float): Convergence threshold for objective change. Default: 1e-6.
            early_stop_delta_threshold (float, optional): Delta threshold for early stopping.
                Defaults to cri.
            early_stop_patience (int): Consecutive decreasing steps before trend reversal
                detection. Default: 3.
            early_stop_patience_no_improve (int): Iterations without improvement before
                stopping. Default: 50.
            early_stop_enabled (bool): Whether to enable early stopping. Default: True.

        Returns:
            tuple: (U, E, V, M, R_matrix)
                U (np.ndarray): Student latent factors (T x M).
                E (np.ndarray): Exercise characteristics (N x T).
                V (np.ndarray): Knowledge requirements (T x K).
                M (np.ndarray): Exercise difficulty parameters (N x 1).
                R_matrix (np.ndarray): Learned prerequisite strengths (K x K),
                                       column-normalized: sum_i R[i, j] = 1.
        """
        if early_stop_delta_threshold is None:
            early_stop_delta_threshold = cri

        u, e, v, m, r_matrix = ae_nmcfg_lipschitz.fit_data(
            self.train_data, self.train_fill, self.Q, self.W, self.rank,
            self.adj_matrix, self.lambda_reg, self.beta_reg,
            max_iter, cri,
            early_stop_delta_threshold=early_stop_delta_threshold,
            early_stop_patience=early_stop_patience,
            early_stop_patience_no_improve=early_stop_patience_no_improve,
            early_stop_enabled=early_stop_enabled)

        return u, e, v, m, r_matrix
