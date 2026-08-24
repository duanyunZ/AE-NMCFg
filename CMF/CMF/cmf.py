# -*- coding: utf-8 -*-
"""
the main module of the CMF framework
"""

import numpy as np
import math

MAX_ITER = 500  # The maximum number of iterations

def fit_data_grad(stu_exe, q_m, rank, alpha, gamma, lamb, beta):
    """ parameter optimizaiton based on the gradient-based method """

    STEP = 0.001
    
    print("[Gradient] Fitting data...")

    stu_num, exe_num = stu_exe.shape
    kno_num = q_m.shape[1]

    # Weight matrix for scoring matrix
    W = np.ones(shape=stu_exe.shape)  # initialization
    W[np.isnan(stu_exe)] = 0

    fun = np.frompyfunc(lambda x:1/(1 + np.exp(-x)), 1, 1)

    # initialize U, V, and Z
    U = np.random.uniform(0, 1, size=(stu_num, rank))
    V = np.random.uniform(0, 1, size=(exe_num, rank))
    Z = np.random.uniform(0, 1, size=(kno_num, rank))

    # calculate the objective function value 
    obj_val = cal_obj(stu_exe, q_m, W, U, V, Z, alpha, gamma, lamb, beta)

    # ---- iterate ----

    convergence = False
    i = 0
    print("Iteration %d = %s" % (i, obj_val))

    while (not convergence) and (i < MAX_ITER):
        """ calculate the next parameter values """
        grad_U = alpha * np.dot(W*(np.asarray(fun(np.dot(U, V.T)))-stu_exe), V) + lamb * U
        grad_V = alpha * np.dot((W*(np.asarray(fun(np.dot(U, V.T)))-stu_exe)).T, U) \
               + (1-alpha) * np.dot(np.asarray(fun(np.dot(V, Z.T)))-q_m, Z) \
               + gamma * V
        grad_Z = (1-alpha) * np.dot((np.asarray(fun(np.dot(V, Z.T)))-q_m).T, V) + beta * Z
        _U = U - STEP * grad_U
        _V = V - STEP * grad_V
        _Z = Z - STEP * grad_Z

        _U, _V, _Z = _U.astype(np.float64), _V.astype(np.float64), _Z.astype(np.float64)
    
        """ calculate the next objective function value """
        obj_val_new = cal_obj(stu_exe, q_m, W, _U, _V, _Z, alpha, gamma, lamb, beta)

        """ update U, V, and Z """
        U = _U
        V = _V
        Z = _Z

        if obj_val_new - obj_val <= 0.01 * sum(sum(grad_U * (_U-U))):
            is_continue = False
        else:
            STEP = STEP / 2  # shrink the step size

        """ is convergent? """
        convergence = abs(obj_val - obj_val_new) / obj_val < 0.05
        obj_val = obj_val_new  # update the objective function value

        i += 1
        print("Iteration %d = %s" % (i, obj_val))
        if i == MAX_ITER:
            print('Maximum iterations reached.')

    return U, V, Z


def cal_obj(stu_exe, q_m, W, U, V, Z, alpha, gamma, lamb, beta):
    L1 = - sum(
                sum(
                    W * stu_exe * np.log(np.where(np.dot(U, V.T) > 0, np.dot(U, V.T), 1e-5))
                    + 
                    W * (1-stu_exe) * np.log(np.where(1-np.dot(U, V.T) > 0, 1-np.dot(U, V.T), 1e-5))
                )
            ) \
         - np.trace(np.dot((W*stu_exe).T, np.dot(U, V.T))) \
         + lamb * math.pow(np.linalg.norm(U, ord='fro'), 2) \
         + gamma * math.pow(np.linalg.norm(V, ord='fro'), 2)
    L2 = - sum(
                sum(
                    q_m * np.log(np.where(np.dot(V, Z.T) > 0, np.dot(V, Z.T), 1e-5))
                    + 
                    (1-q_m) * np.log(np.where(1-np.dot(V, Z.T) > 0, 1-np.dot(V, Z.T), 1e-5))
                )
            ) \
         - np.trace(np.dot(q_m.T, np.dot(V, Z.T))) \
         + beta * math.pow(np.linalg.norm(Z, ord='fro'), 2)
    obj_val = alpha * L1 + (1-alpha) * L2
    return obj_val



def fit_data_newton(stu_exe, q_m, rank, alpha, gamma, lamb, beta):
    """ parameter optimizaiton based on the Newton method """

    STEP = 0.001

    print("[Newton] Fitting data...")

    stu_num, exe_num = stu_exe.shape
    kno_num = q_m.shape[1]

    # Weight matrix for scoring matrix
    W = np.ones(shape=stu_exe.shape)  # initialization
    W[np.isnan(stu_exe)] = 0

    fun = np.frompyfunc(lambda x:1/(1 + np.exp(-x)), 1, 1)
    fun_prime = np.frompyfunc(lambda x: (1/(1 + np.exp(-x)))*(1-1/(1 + np.exp(-x))), 1, 1)

    # initialize U, V, and Z
    U = np.random.uniform(0, 1, size=(stu_num, rank))
    V = np.random.uniform(0, 1, size=(exe_num, rank))
    Z = np.random.uniform(0, 1, size=(kno_num, rank))

    # calculate the objective function value 
    obj_val = cal_obj(stu_exe, q_m, W, U, V, Z, alpha, gamma, lamb, beta)

    # ---- iterate ----

    convergence = False
    i = 0
    print("Iteration %d = %s" % (i, obj_val))

    while (not convergence) and (i < 100):
        """ calculate the next parameter values """
        
        _U = np.zeros(shape=(stu_num, rank))
        _V = np.zeros(shape=(exe_num, rank))
        _Z = np.zeros(shape=(kno_num, rank))

        grad_U = alpha * np.dot(W*(np.asarray(fun(np.dot(U, V.T)))-stu_exe), V) + lamb * U
        grad_V = alpha * np.dot((W*(np.asarray(fun(np.dot(U, V.T)))-stu_exe)).T, U) \
               + (1-alpha) * np.dot(np.asarray(fun(np.dot(V, Z.T)))-q_m, Z) \
               + gamma * V
        grad_Z = (1-alpha) * np.dot((np.asarray(fun(np.dot(V, Z.T)))-q_m).T, V) + beta * Z
        grad_U, grad_V, grad_Z = grad_U.astype(np.float64), grad_V.astype(np.float64), grad_Z.astype(np.float64)
        
        for m in range(stu_num):
            q_u = grad_U[m,:]
            D1 = W[m,:] * np.asarray(fun_prime(np.dot(U[m,:],V.T)))
            D1 = np.diag(D1.astype(np.float64))
            q_u_prime = alpha * np.dot(np.dot(V.T, D1), V) + np.diag([lamb]*rank)
            _U[m,:] = U[m,:] - STEP * np.dot(q_u, 1/q_u_prime)
        
        for n in range(exe_num):
            q_v = grad_V[n,:]
            D2 = W[:,n] * np.asarray(fun_prime(np.dot(U,V[n,:].T)))
            D2 = np.diag(D2.astype(np.float64))
            D3 = np.asarray(fun_prime(np.dot(V[n,:],Z.T)))
            D3 = np.diag(D3.astype(np.float64))
            q_v_prime = alpha * np.dot(np.dot(U.T, D2), U) + (1-alpha) * np.dot(np.dot(Z.T,D3),Z) + np.diag([gamma]*rank)
            _V[n,:] = V[n,:] - STEP * np.dot(q_v, 1/q_v_prime)

        for k in range(kno_num):
            q_z = grad_Z[k,:]
            D4 = np.asarray(fun_prime(np.dot(V,Z[k,:].T)))
            D4 = np.diag(D4.astype(np.float64))
            q_z_prime = (1-alpha) * np.dot(np.dot(V.T,D4),V) + np.diag([beta]*rank)
            _Z[k,:] = Z[k,:] - STEP * np.dot(q_z, 1/q_z_prime)
        
        """ calculate the next objective function value """
        obj_val_new = cal_obj(stu_exe, q_m, W, _U, _V, _Z, alpha, gamma, lamb, beta)

        """ update U, V, and Z """
        U = _U
        V = _V
        Z = _Z

        if obj_val_new - obj_val <= 0.01 * sum(sum(grad_U * (_U-U))):
            is_continue = False
        else:
            STEP = STEP / 2  # shrink the step size  

        """ is convergent? """
        convergence = abs(obj_val - obj_val_new) < 0.1
        obj_val = obj_val_new  # update the objective function value

        i += 1
        print("Iteration %d = %s" % (i, obj_val))
        
    return U, V, Z