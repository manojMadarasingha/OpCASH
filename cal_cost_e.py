# caclculate the cost bw
import os
import pandas as pd
import numpy as np
import time


# store the matrix A
def store_bw_e2u(df_ct, store_path, n):
    user_path = store_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df_ct.to_csv(user_path + '/cost_e.csv', index=False)

    return


# generate the cost_e vector for a given set of cached tiles of a given user n
# calculate the DT in BT sizes and store them
# we user the derive eq:2 in the paper.
def generate_cost_e(cached_tiles_m, bt_size_arr, n,
                    data_store_path, ena_store):
    start_time = time.time()

    all_dt_sizes = []
    for t_ind, t in enumerate(cached_tiles_m):

        # get sum of basic tiles
        tot_bts_size = 0
        l_l_m = int(t[0])
        l_l_n = int(t[1])
        u_r_m = int(t[2])
        u_r_n = int(t[3])
        for r in range(l_l_m, u_r_m):
            for c in range(l_l_n, u_r_n):
                bt_ind = r * 20 + c
                tot_bts_size += bt_size_arr[bt_ind]
        tot_bts_size = tot_bts_size / 1000000
        dt_size_mb = 0.432 * tot_bts_size * tot_bts_size + 0.306 * tot_bts_size + 0.0025
        all_dt_sizes.append(dt_size_mb)

    stop_time = time.time()

    # store the data
    ct_col_name = np.repeat(['ct_'], len(all_dt_sizes))
    cts = np.arange(len(all_dt_sizes)).astype(str)
    ct_cols = np.core.defchararray.add(ct_col_name, cts)

    df_cost_bw_e2u = pd.DataFrame(columns=ct_cols,
                                  data=np.asarray(all_dt_sizes).reshape([1, -1]))

    if ena_store:
        store_bw_e2u(df_cost_bw_e2u, data_store_path, n)

    return np.asarray(all_dt_sizes), stop_time - start_time
