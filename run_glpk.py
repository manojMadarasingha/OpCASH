# this funciton run the glpk based ILP solution

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import time
import glpk  # Import the GLPK module


# create store the indices
def store_indicies(index_values, data_store_path, n):
    ct_col_name = np.repeat(['ct_'], len(index_values))
    cts = np.arange(len(index_values)).astype(str)
    ct_cols = np.core.defchararray.add(ct_col_name, cts)

    df_iou = pd.DataFrame(columns=ct_cols,
                          data=np.asarray(index_values).reshape([1, -1]))

    df_iou.to_csv(data_store_path + '/user_' + str(n) + '/ct_indices.csv', index=False)

    return


def store_sel_ct(sel_ct, data_store_path, n):
    columns = ['l_l_m', 'l_l_n', 'u_r_m', 'u_r_n', 'n', 'u', 'id', 'hit']
    if len(sel_ct) > 0:
        data = sel_ct
        df = pd.DataFrame(columns=columns,
                          data=data)
    else:
        df = pd.DataFrame(columns=columns)
    df.to_csv(data_store_path + '/user_' + str(n) + '/tiles_cts.csv', index=False)

    return


# create the object function
def create_obj_function(cost_r, cost_e, cost_s,
                        w1, w2, w3):
    # if we find valid tiles in that can be cached from the MEC
    if len(cost_r) > 1:
        cost_view = cost_r
        # cost_bw = (w1 * cost_e) + (1 - beta) * (-1 * cost_s)
        cost_bw = w2 * cost_e - w3 * cost_s

    # if not there are valid tiles to be fetched from the MEC
    else:
        cost_view = cost_r

        # validate cost_e for <1 values
        if cost_e[0] <= 1:
            cost_bw = w1 * cost_e
        else:
            cost_bw = 0

        # validate cost_s for <1 values
        if cost_s[0] <= 1:
            cost_bw += (w3 * cost_s)
        else:
            cost_bw += 0

    cost_func = w1 * cost_view + cost_bw

    return cost_func


# run the glpk (ILP solution) to find the optimal cached tile coverage based on the cost functions
# cost_r, cost_e and cost_s
def run_glpk(cost_r, cost_e, cost_s, A,
             w1, w2, w3):
    lp = glpk.LPX()  # Create empty problem instance

    lp.name = 'sample'  # Assign symbolic name to problem
    lp.obj.maximize = True  # Set this as a maximization problem

    lp.rows.add(A.shape[0])  # Append three rows to this instance
    for r in lp.rows:
        r.name = 'p' + str(r)  # Name them p, q, and r
    for i in range(A.shape[0]):
        lp.rows[i].bounds = 0, 1  # Set bound -inf < p <= 100

    lp.cols.add(A.shape[1])  # Append three columns to this instance
    for c in lp.cols:  # Iterate over all columns
        c.name = 'x%d' % c.index  # Name them x0, x1, and x2
        c.bounds = (0, 1)  # Set bound 0 <= xi < inf

    cost_func = create_obj_function(cost_r, cost_e, cost_s,
                                    w1, w2, w3)

    y = list(cost_func.reshape([1, -1])[0])
    lp.obj[:] = y  # Set objective coefficients

    A_flat = list(A.flatten())

    lp.matrix = A_flat
    lp.simplex()  # Solve this LP with the simplex method
    # print('Z = %g;' % lp.obj.value)  # Retrieve and print obj func value
    # print('; '.join('%s = %g' % (c.name, c.primal) for c in lp.cols))

    index_vals = []
    for c in lp.cols:
        index_vals.append(round(c.primal))

    return index_vals


def get_ilp_based_sol(A,
                      cost_r, cost_e, cost_s,
                      n,
                      data_store_path,
                      w1, w2, w3,
                      ct,
                      ena_store):
    start_time = time.time()

    if cost_r.shape[1] > 1:
        cost_r = np.squeeze(cost_r)
    else:
        cost_r = cost_r[0, :]

    # normalize only if the iou has more than one value
    if cost_r.shape[0] > 1:
        cost_r = (cost_r - np.min(cost_r)) / (np.max(cost_r) - np.min(cost_r))
        cost_e = (cost_e - np.min(cost_e)) / (np.max(cost_e) - np.min(cost_e))
        cost_s = (cost_s - np.min(cost_s)) / (np.max(cost_s) - np.min(cost_s))

    A = A[:, 2:]

    cost_r = np.nan_to_num(cost_r)
    cost_e = np.nan_to_num(cost_e)
    cost_s = np.nan_to_num(cost_s)

    index_values = run_glpk(cost_r, cost_e, cost_s, A,
                            w1, w2, w3)

    sel_ind = np.where(np.asarray(index_values) > 0)
    ct = np.asarray(ct)
    sel_ct = ct[sel_ind, :][0]

    stop_time = time.time()

    if ena_store:
        store_indicies(index_values, data_store_path, n)
        store_sel_ct(sel_ct, data_store_path, n)

    return sel_ct, stop_time - start_time
