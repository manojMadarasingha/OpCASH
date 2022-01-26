import pandas as pd
import numpy as np
import time
import os

# get the basic tiles covering a given DT tile
def get_bt_dist(tiles):
    bts = np.zeros((10, 20))
    for t in tiles:
        bts[int(t[0]):int(t[2]), int(t[1]):int(t[3])] = 1

    return bts


# get the cached tiles overlapped with the user vp
def get_ct_on_vp(vp_bt,cached_tiles):
    overlapped_tiles = []
    for ct in cached_tiles:
        if np.sum(vp_bt[int(ct[0]):int(ct[2]), int(ct[1]):int(ct[3])]) > 0:
            overlapped_tiles.append(ct)

    return overlapped_tiles


# store the selected tiles
def store_sel_tiles(ct_sel, store_path, n):

    columns = ['l_l_m', 'l_l_n', 'u_r_m', 'u_r_n', 'n', 'u', 'id', 'hit']

    if len(ct_sel)>0:
        ct_sel = np.asarray(ct_sel)
        df = pd.DataFrame(columns=columns,
                          data=ct_sel)
    else:
        df = pd.DataFrame(columns=columns)
    user_path = store_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df.to_csv(user_path + '/overlapped_cts.csv', index=False)

    return


# store the matrix A
def store_matrix_A(df_ct, store_path, n):
    user_path = store_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df_ct.to_csv(user_path + '/A.csv', index=False)

    return


def create_matrix_A(vp, n, store_path,cached_tiles,ena_store):
    start_time = time.time()

    # get bt distribution of vps
    vp_bt = get_bt_dist(vp)

    # get the overlapping cached tiles on the vp bts
    ct_sel = get_ct_on_vp(vp_bt,cached_tiles)

    if len(ct_sel) > 0:
        # get the bt coverage from the ct
        ct_bt = get_bt_dist(ct_sel)

        # create final bt cover overlpping both vp_bt and ct_bt
        final_bt = np.logical_or(vp_bt, ct_bt)

        A = np.zeros((int(np.sum(final_bt)), int(len(ct_sel))))
        bt_indices = np.swapaxes(np.asarray(np.where(final_bt == 1)), 0, 1)

        cols = ['r', 'c']
        ct_name = np.repeat(['ct_'], len(ct_sel))
        cts = np.arange(len(ct_sel)).astype(str)
        ct_cols = np.core.defchararray.add(ct_name, cts)
        cols.extend(ct_cols)

        df_A = pd.DataFrame(columns=cols,
                            data=np.concatenate([bt_indices, A], axis=1))

        for ct_ind, c_t in enumerate(ct_sel):
            ct_cover = np.zeros((10, 20))
            ct_cover[int(c_t[0]):int(c_t[2]), int(c_t[1]):int(c_t[3])] = 1
            bt_ct_overlap = np.where((ct_cover + final_bt) == 2)

            if len(bt_ct_overlap[0]) > 0:
                r = bt_ct_overlap[0].astype(str)
                c = bt_ct_overlap[1].astype(str)
                r_true = df_A.r.isin(r)
                c_true = df_A.c.isin(c)
                df_A.loc[r_true & c_true, 'ct_' + str(ct_ind)] = 1
    else:
        bt_indices = np.swapaxes(np.asarray(np.where(vp_bt == 1)), 0, 1)
        cols = ['r', 'c']
        df_A = pd.DataFrame(columns=cols,
                            data=bt_indices)

    # stop measuring the time return to the main function
    stop_time = time.time()

    if ena_store:
        # store the cached tiles after every user
        store_sel_tiles(ct_sel, store_path, n)
        # store the user matrix A after each user
        store_matrix_A(df_A, store_path, n)

    return df_A.values, stop_time - start_time, ct_sel