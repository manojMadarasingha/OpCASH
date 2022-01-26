# This python script combine all the functional blocks with the improved version
# consider only the cached tiles of the previous users
# dynamic matrix A creation

import pandas as pd
import numpy as np
import os

import settings
import create_mat_A
import cal_cost_r
import cal_cost_e
import cal_cost_s
import run_glpk
import partition_remianing_area
import check_distribution
import further_partition_new_tiles
import multiprocessing as mp

is_DEBUG = True


# ++++++++++++++++++++++++++++#
# cache replace only if the number of users are greater than 15
def cache_replace(cached_tiles, n):
    cache_user_margin = 15
    cache_back_usage = 10
    cache_removing_percentile = 0.2
    cache_expected_hit_prob_of_removing_tiles = 0.2

    print(cached_tiles)
    ch_replace_range = cached_tiles[cached_tiles[:, 4] < (n - cache_back_usage)]
    ch_non_replace_range = cached_tiles[cached_tiles[:, 4] >= (n - cache_back_usage)]

    # find the indicies should be removed
    # hit_val = ch_replace_range[:,-1]
    # hit_val_prob = hit_val/np.sum(hit_val)
    print(ch_replace_range)
    ch_replace_range = ch_replace_range[np.lexsort((-ch_replace_range[:, 4], -ch_replace_range[:, 7]))]
    # cut the last 20% of the tiles
    last_20_ind = int(len(ch_replace_range) * cache_removing_percentile)
    total_hits = np.sum(ch_replace_range[:, 7])
    prob_cachehit_las_20 = np.sum(ch_replace_range[-last_20_ind:, 7]) / total_hits
    if prob_cachehit_las_20 < cache_expected_hit_prob_of_removing_tiles:
        ch_replace_range = ch_replace_range[:int(len(ch_replace_range) * (1 - cache_removing_percentile)), :]
        # ch_replace_range = ch_replace_range[np.lexsort(ch_replace_range[:, 4])]
        ch_replace_range = ch_replace_range[ch_replace_range[:, 4].argsort()]
    cached_tiles = np.concatenate([ch_replace_range, ch_non_replace_range], axis=0)

    return cached_tiles


def update_hits_in_cached_tiles(hit_cts, cached_tiles):
    y = hit_cts[:, 6]
    for i in y:
        indices = np.argwhere(cached_tiles[:, 6] == i)
        cached_tiles[indices, 7] += 1
    return cached_tiles


# Taking a given set of tiles from a user has streamed
# find the tiles areaday has been cached and store the new tiles from CS to be cached.
# cached the tiles into the global list with the user n data
# This cached tile list is a Global list that all users can see
def fill_cached_tiles(tiles, n, u, cached_tiles, hit_cts):
    if len(tiles) == 0:
        cached_tiles = update_hits_in_cached_tiles(hit_cts, cached_tiles)
    else:
        # add last column indicating the tile requesting order and the user order
        if n == 0:
            last_ind = -1
        else:
            last_ind = cached_tiles[-1, 6]

        new_indices = np.arange(last_ind + 1, len(tiles) + last_ind + 1).reshape([-1, 1])
        hits = np.zeros(len(tiles)).reshape([-1, 1])

        request_order = np.repeat([n], tiles.shape[0]).reshape([-1, 1])
        user_num = np.repeat([u], tiles.shape[0]).reshape([-1, 1])
        new_arr = np.concatenate([tiles, request_order, user_num, new_indices, hits], axis=1)

        if n == 0:
            cached_tiles = np.asarray(new_arr)
        else:
            cached_tiles = np.concatenate([cached_tiles, new_arr], axis=0)

    return cached_tiles


def store_all_cached_tiles(cached_tiles, n, storage_path):
    ct_sel = cached_tiles
    columns = ['l_l_m', 'l_l_n', 'u_r_m', 'u_r_n', 'n', 'u', 'id', 'hit']
    df = pd.DataFrame(columns=columns,
                      data=ct_sel)
    user_path = storage_path + '/user_' + str(n)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    df.to_csv(user_path + '/total_cts.csv', index=False)

    return


def remove_repeating_tiles(vp_tiles):
    vp_tiles = list(vp_tiles)
    tiles = []
    for t in vp_tiles:
        if not list(t) in tiles:
            tiles.append(list(t))

    return np.asarray(tiles)


def run_algo(vid, chunk, rand_set,
             w1, w2, w3, bw,
             DT_path, BT_path, Data_store_path):
    # array to store the time consumption for the key tasks of the process.
    t_for_complete = []

    # read random sets for the users are in different order.
    random_sets = settings.rand_set
    cached_tiles = []

    # define paths for inputs and outputs
    vid_DT_path_in = DT_path + '/' + vid
    BT_size_path = BT_path + '/' + vid
    data_store_path = Data_store_path + '/' + vid + '/Random_set' + str(rand_set) + '/chunk_' + str(chunk)

    if not os.path.exists(data_store_path):
        os.makedirs(data_store_path)

    # read the bt size for the given chunk
    BT_size = pd.read_csv(BT_size_path + '.csv').values

    for n, u in enumerate(random_sets[rand_set]):
        print('user' + str(n))

        # read the DT tiles of individual users
        user_path = vid_DT_path_in + '/user_' + str(u) + '/chunk_' + str(chunk) + '.csv'
        DT_tiles = pd.read_csv(user_path).values[:, :-1]

        # remove any repeating tiles if there is
        DT_tiles = remove_repeating_tiles(DT_tiles)

        # take all the tiles streamed by user n=0 from the CS as the cached tiles. n==0 means the first user
        if n == 0:
            cached_tiles = fill_cached_tiles(DT_tiles, n, u, cached_tiles, hit_cts=[])
            store_all_cached_tiles(cached_tiles, n, data_store_path)
        # run the algorithm for the users n>=1
        else:

            # create matrix A. Also find wehether there is any cached tiles overlapping with the tiles
            A, time_pre_A, overlapped_ct = create_mat_A.create_matrix_A(DT_tiles, n,
                                                                        data_store_path,
                                                                        cached_tiles,
                                                                        ena_store=True)
            t_for_complete.append(time_pre_A)

            # for the remaining users, we first find whether there is any overlapped tiles with the VP of the user n
            if len(overlapped_ct) > 0:

                # generate the cost r
                cost_r, time_cost_r = cal_cost_r.generate_cost_r(DT_tiles, n,
                                                                 data_store_path,
                                                                 overlapped_ct,
                                                                 ena_store=True)
                t_for_complete.append(time_cost_r)

                # calculate the cost_e
                cost_e, time_cost_e = cal_cost_e.generate_cost_e(overlapped_ct,
                                                                 BT_size[chunk, 1:],
                                                                 n,
                                                                 data_store_path,
                                                                 ena_store=True)
                t_for_complete.append(time_cost_e)

                # calculate the cost_s
                cost_s, time_for_cost_s = cal_cost_s.generate_cost_s(DT_tiles,
                                                                     overlapped_ct,
                                                                     BT_size[chunk, 1:],
                                                                     n,
                                                                     data_store_path,
                                                                     ena_store=True)
                t_for_complete.append(time_for_cost_s)

                # cost latency factor is calculated to measure the time for
                # tile streaming between the core and the last mile.
                # if bw != 0:
                #     cost_lat_s2e = cost_s / bw
                #     cost_lat_s2e = np.nan_to_num(cost_lat_s2e)
                #     cost_lat_s2e = np.nan_to_num(cost_lat_s2e, posinf=0)
                # else:
                #     cost_lat_s2e = np.zeros(len(cost_s))

                # run glpk. Return the selected tiles with their coordinates
                sel_ct, t_ilp_sol = run_glpk.get_ilp_based_sol(A,
                                                               cost_r, cost_e, cost_s,
                                                               n,
                                                               data_store_path,
                                                               w1, w2, w3,
                                                               overlapped_ct,
                                                               ena_store=True)
                # print(sel_ct)
                t_for_complete.append(t_ilp_sol)

                # once we detect the tiles to from the cached, to cover the remaining parts of the VP, we fetch tiles from CS
                # These remaining parts might not be perfect rectangles. We run VASTile implementaion further to further
                # partition the tiles. This process should be done at the CS.
                # functions returns the new tiles to be fetchd from the CS
                fetch_t, t_fetch_new_t = partition_remianing_area.find_new_tiles_to_request(DT_tiles, sel_ct,
                                                                                            data_store_path, n, u,
                                                                                            video_name=vid,
                                                                                            f=chunk,
                                                                                            gamma=1)
                t_for_complete.append(t_fetch_new_t)

                # --------------Part of VASTile implementation for furthe partitioning the tiles starts -------------- #
                # further partitioned the tiles
                fetch_t_further_partitioned, t_fetch_new_furth_part_t = further_partition_new_tiles.partition_new_tiles(
                    fetch_t, data_store_path, n, u, gamma=1, ena_store=True)
                t_for_complete.append(t_fetch_new_furth_part_t)
                # --------------Part of VASTile implementation for furthe partitioning the tiles ends -------------- #

                # fill the cached hit analysis
                if len(sel_ct) > 0:
                    cached_tiles = fill_cached_tiles([], n, u, cached_tiles,
                                                     hit_cts=sel_ct)
                # update the catched tiles witht the newly fetched tiles
                if len(fetch_t) > 0:
                    cached_tiles = fill_cached_tiles(np.asarray(fetch_t_further_partitioned), n, u, cached_tiles,
                                                     hit_cts=[])
                store_all_cached_tiles(cached_tiles, n, data_store_path)

                # If Debug is enabled, plot the graphs at the 20th user. Change this number to see the tiles
                # green: Original VP
                # red: Optimal tile cover from the cache
                #
                if is_DEBUG:
                    if chunk == 20:
                        check_distribution.draw_tiles(DT_tiles, sel_ct, fetch_t_further_partitioned, data_store_path, n)
            # if we can not find any overlapped tiles, then simply store only the tiles_new.csv with the vp tiles
            # no cache tiles to be stored.
            else:
                further_partition_new_tiles.store_new_t(DT_tiles, data_store_path, n, u)
                run_glpk.store_sel_ct([], data_store_path, n)
                cached_tiles = fill_cached_tiles(np.asarray(DT_tiles), n, u, cached_tiles, hit_cts=[])
                store_all_cached_tiles(cached_tiles, n, data_store_path)

    return


def run_opcash(vid, w1, w2, w3, bw_trace, DT_path, BT_path, Data_store_path):
    # Run different random set of user sequence.
    # maximum there are 3 randomized orders in settings.py
    for rand_set in range(1):
        for chunk in range(120):
            run_algo(vid, chunk, rand_set,
                     w1, w2, w3,
                     bw_trace[int(chunk // 2)],
                     DT_path, BT_path, Data_store_path)

    return


def main():
    work_dir = os.getcwd()
    DT_path = work_dir + '/Tile_info_DT'
    BT_path = work_dir + '/Tile_info_BT'
    Data_store_path = work_dir + '/Data_store_path'

    bw_trace = pd.read_csv(work_dir + '/BW_traces/4G.csv').values[6, :-1]
    bw_trace = bw_trace / 8

    w1 = 0.6
    w2 = 0.25
    w3 = 0.15

    vids = settings.vid_test

    for v, vid in enumerate(vids):
        run_opcash(vid, w1, w2, w3, bw_trace, DT_path, BT_path, Data_store_path)


if __name__ == main():
    main()
