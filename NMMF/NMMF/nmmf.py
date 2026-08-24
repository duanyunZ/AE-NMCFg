# -*- coding: utf-8 -*-
"""
the main module of the NMMF framework
"""

import numpy as np

MAX_ITER = 500  # The maximum number of iterations
CRI = 0.5

def fit_data(stu_exe, q_m, rank, beta):

    print("Fitting data...")

    exe_num, stu_num = np.shape(stu_exe)
    kno_num = len(q_m[1])

    # initialize B, W, and H
    b = np.random.uniform(0, 1, size=(rank, kno_num))
    w = np.random.uniform(0, 1, size=(exe_num, rank))
    h = np.random.uniform(0, 1, size=(rank, stu_num))

    # calculate the objective function value
    obj_val = _div(stu_exe, w, h) + beta*_div(q_m, w, b)
    
    # ---- iterate ----

    convergence = False
    i = 0
    print("Iteration %d = %s" % (i, obj_val))

    while (not convergence) and (i < MAX_ITER):
        """ calculate the next parameter values """
        _b = update_b(q_m, w, b)
        _w = update_w(stu_exe, q_m, _b, w, h, beta)
        _h = update_h(stu_exe, _w, h)

        """ calculate the next objective function value """
        obj_val_new = _div(stu_exe, _w, _h) + beta*_div(q_m, _w, _b)

        """ update B, W, and H """
        b = _b
        w = _w
        h = _h

        """ is convergent? """
        convergence = abs(obj_val - obj_val_new) < CRI
        obj_val = obj_val_new  # update the objective function value

        i += 1
        print("Iteration %d = %s" % (i, obj_val))
        if i == MAX_ITER:
            print('Maximum iterations reached.')
        
    return b, w, h


def _div(true_m, l_m, r_m):
    row, col = np.shape(true_m)
    div_val = 0
    for i in range(row):
        for j in range(col):
            if true_m[i,j] == 0:
                continue
            div_val += true_m[i,j] * np.log(true_m[i,j]/np.dot(l_m[i,:],r_m[:,j])) - true_m[i,j] - np.dot(l_m[i,:],r_m[:,j])
    return div_val


def update_b(q_m, w, b):
    b_new = np.zeros(shape=b.shape)
    rank = w.shape[1]
    exe_num, kno_num = q_m.shape
    for k in range(rank):
        for m in range(kno_num):
            nume, deno = 0, 0
            for i in range(exe_num):
                if not q_m[i,m] == 0:
                    nume += q_m[i,m]/np.dot(w[i,:],b[:,m]) * w[i,k]
                deno += w[i,k]
            b_new[k,m] = b[k,m] * (nume/deno)
    return b_new


def update_w(stu_exe, q_m, b, w, h, beta):
    w_new = np.zeros(shape=w.shape)
    exe_num, rank = w.shape
    stu_num = stu_exe.shape[1]
    kno_num = q_m.shape[1]
    for i in range(exe_num):
        for k in range(rank):
            nume, deno = 0, 0
            for j in range(stu_num):
                if not stu_exe[i,j] == 0:
                    nume += (stu_exe[i,j]/np.dot(w[i,:],h[:,j])) * h[k,j]
                deno += h[k,j]
            for m in range(kno_num):
                if not q_m[i,m] == 0:
                    nume += beta * (q_m[i,m]/np.dot(w[i,:],b[:,m])) * b[k,m]
                deno += beta * b[k,m]
            w_new[i,k] = w[i,k] * (nume/deno)
    return w_new


def update_h(stu_exe, w, h):
    h_new = np.zeros(shape=h.shape)
    exe_num = stu_exe.shape[0]
    rank, stu_num = h.shape
    for k in range(rank):
        for j in range(stu_num):
            nume, deno = 0, 0
            for i in range(exe_num):
                if not stu_exe[i,j] == 0:
                    nume += (stu_exe[i,j]/np.dot(w[i,:],h[:,j])) * w[i,k]
                deno += w[i,k]
            h_new[k,j] = h[k,j] * (nume/deno)
    return h_new