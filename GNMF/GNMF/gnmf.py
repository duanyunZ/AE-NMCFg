# -*- coding: utf-8 -*-
"""
the main module of the GNMF framework
"""

import numpy as np
import math

MAX_ITER = 500  # The maximum number of iterations
CRI = 10
L = 2 # the number of matrix components
C = 2


def fit_data(stu_exe, q_m, rank, gamma, alpha, beta, lamb=1):

    print("Fitting data...")

    exe_num, stu_num = stu_exe.shape
    kno_num = q_m.shape[1]
    X = []
    X.append(stu_exe)
    X.append(q_m)

    # initialize A and S
    A, S = [], []
    A.append(np.random.uniform(0, 1, size=(exe_num, rank)))
    A.append(np.random.uniform(0, 1, size=(exe_num, rank)))
    S.append(np.random.uniform(0, 1, size=(rank, stu_num)))
    S.append(np.random.uniform(0, 1, size=(rank, kno_num)))

    # calculate the objective function value
    obj_val = 0
    for l in range(L):
        obj_val += lamb * math.pow(np.linalg.norm(X[l]-np.dot(A[l],S[l]), ord='fro'), 2) \
                + gamma * math.pow(np.linalg.norm(A[l], ord='fro'), 2)
        delta_c, delta_i = 0, 0
        for j in range(L):
            if not j == l:
                delta_c += 0.5 * alpha * math.pow(np.linalg.norm(A[j][:,0:C]-A[l][:,0:C], ord='fro'), 2)
                delta_i += 0.5 * beta * math.pow(np.linalg.norm(A[j][:,C:]-A[l][:,C:], ord='fro'), 2)
        obj_val += delta_c - delta_i
    
    # ---- iterate ----
    convergence = False
    i = 0
    print("Iteration %d = %s" % (i, obj_val))

    while (not convergence) and (i < MAX_ITER):
        """ calculate the next parameter values """
        _S = update_S(X, A, S)
        _A = update_A(X, A, _S, rank, lamb, gamma, alpha, beta)

        """ calculate the next objective function value """
        obj_val_new = 0
        obj_val_new += lamb * math.pow(np.linalg.norm(X[l]-np.dot(_A[l],_S[l]), ord='fro'), 2) \
                    + gamma * math.pow(np.linalg.norm(_A[l], ord='fro'), 2)
        delta_c_new, delta_i_new = 0, 0
        for j in range(L):
            if not j == l:
                delta_c_new += 0.5 * alpha * math.pow(np.linalg.norm(_A[j][:,0:C]-_A[l][:,0:C], ord='fro'), 2)
                delta_i_new += 0.5 * beta * math.pow(np.linalg.norm(_A[j][:,C:]-_A[l][:,C:], ord='fro'), 2)
        obj_val_new += delta_c_new - delta_i_new

        """ update A and S """
        S = _S
        A = _A

        """ is convergent? """
        convergence =  obj_val_new < 0
        obj_val = obj_val_new  # update the objective function value

        i += 1
        print("Iteration %d = %s" % (i, obj_val))
        if i == MAX_ITER:
            print('Maximum iterations reached.')

    return A, S


def update_S(X, A, S):
    S_new = []
    for l in range(L):
        S_new.append(S[l] * (
                        (np.dot(A[l].T,X[l]))
                        /
                        (np.dot(np.dot(A[l].T,A[l]), S[l]))
                        )
                )
    bias = 0.0005
    for l in range(L):
        S_new[l] = S_new[l] + bias
    return S_new

def update_A(X, A, S, rank, lamb, gamma, alpha, beta):
    A_new = []
    A_new.append(np.zeros(shape=(X[0].shape[0], rank)))
    A_new.append(np.zeros(shape=(X[0].shape[0], rank)))

    for l in range(L):
        
        delta_c, delta_i = 0, 0
        for j in range(L):
            if not j == l:
                delta_c += alpha/lamb * A[j][:,0:C]
                delta_i += beta/lamb * A[j][:,C:]
        
        A_new[l][:,0:C] = A[l][:,0:C] * (
            (np.dot(X[l],S[l].T[:,0:C]) +  delta_c)
            /
            (np.dot(np.dot(A[l],S[l]),S[l].T[:,0:C]) + alpha/lamb*(L-1)*A[l][:,0:C] + gamma/lamb*A[l][:,0:C])
        )

        A_new[l][:,C:] = A[l][:,C:] * (
            (np.dot(X[l],S[l].T[:,C:]) + beta/lamb*(L-1)*A[l][:,C:])
            /
            (np.dot(np.dot(A[l],S[l]),S[l].T[:,C:]) + delta_i + gamma/lamb*A[l][:,C:])
        )
    return A_new