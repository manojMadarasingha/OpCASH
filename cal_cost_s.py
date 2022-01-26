# python script cal cost bw s2e
import os
import pandas as pd
import numpy as np
import time


# store the matrix A
def store_bw_s2e(df_ct, store_path, n):
    user_path = store_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df_ct.to_csv(user_path + '/cost_bw_s2e.csv', index=False)

    return


# calculate the DT in BT sizes and store them
def generate_cost_s(vp_tiles, ct, bt_size_arr, n, data_store_path, ena_store):
    # calculate the DT in BT sizes and store them
    start_time = time.time()

    all_dt_sizes = []
    vp_area = np.zeros((10, 20))
    for t in vp_tiles:
        vp_area[int(t[0]):int(t[2]), int(t[1]):int(t[3])] = 1

    # for each tile get the overlap with the user VP area. check the remaining area to be fetched from the
    # server after the coverage of cached tile.
    for t_ind, t in enumerate(ct):

        cached_tile_area = np.zeros((10, 20))
        cached_tile_area[int(t[0]):int(t[2]), int(t[1]):int(t[3])] = 1

        # first get the xor operation between the vp area and the cached tile area
        # completely overlapped or completeley non-overlapped region will be 1
        xor_op = np.logical_xor(vp_area, cached_tile_area)

        # get the element wise multiplication to get the non overlapped region in VP tile
        # with the overlapping region.
        non_overlap_reg = np.multiply(vp_area, xor_op)

        # get the indicies of non-overlapped region
        ind = np.where(non_overlap_reg > 0)

        tot_bts_size = 0
        for i in range(len(ind[0])):
            bt_ind = ind[0][i] * 20 + ind[1][i]
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
        store_bw_s2e(df_cost_bw_e2u, data_store_path, n)

    return np.asarray(all_dt_sizes), stop_time - start_time
