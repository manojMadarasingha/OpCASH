# implement the cost fuctions for the cost view
# cost_iou and cost_dst
import pandas as pd
import numpy as np
import time
import os



# store the matrix A
def store_view(df_ct, store_path, n):
    user_path = store_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df_ct.to_csv(user_path + '/cost_view.csv', index=False)

    return

def create_cost_r(tiles_n, cts):
    start_time = time.time()
    # for each cached tiles calculate the total distance for each tile from the n^th user
    c_view = []
    for ct in cts:
        ct_mat = np.zeros((10, 20))
        # ct_center = [(ct[0] + ct[2]) / 2, (ct[1] + ct[3]) / 2]
        ct_mat[int(ct[0]):int(ct[2]), int(ct[1]):int(ct[3])] = 1
        total_ct_tiles = np.sum(ct_mat)

        for n_tile in tiles_n:
            n_mat = np.zeros((10, 20))
            n_mat[int(n_tile[0]):int(n_tile[2]), int(n_tile[1]):int(n_tile[3])] = 1
            ct_mat -= n_mat

        # measure the cost function with redundnat pixels
        reudndant = np.count_nonzero(ct_mat == 1)
        c_view.append(1-(reudndant / total_ct_tiles))

    # # stop measuring the time return to the main function
    stop_time = time.time()

    return c_view, stop_time - start_time


# generate the cost view
def generate_cost_r(vp_tiles, n, store_path, cached_tiles, ena_store):
    # create matrix A for user n considering the previous cached user detials
    c_view, time = create_cost_r(vp_tiles, cached_tiles)

    ct_col_name = np.repeat(['ct_'], len(c_view))
    cts = np.arange(len(c_view)).astype(str)
    ct_cols = np.core.defchararray.add(ct_col_name, cts)

    df_redunt = pd.DataFrame(columns=ct_cols,
                             data=np.asarray(c_view).reshape([1, -1]))

    if ena_store:
        # store the cached tiles after every user
        store_view(df_redunt, store_path, n)

    return df_redunt.values, time
