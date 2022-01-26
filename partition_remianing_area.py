import numpy as np
import time
import repartition_given_polygon



def find_new_tiles_to_request(vp_tiles, c_tiles, data_store_path, n, u, video_name, f, gamma):
    start_time = time.time()

    tiles_to_fetch = []
    mat_repartition = np.zeros((10, 20))

    # check the overlapping tiles with the c tiles
    for v_t in vp_tiles:
        vp_mat = np.zeros((10, 20))
        vp_mat[int(v_t[0]):int(v_t[2]), int(v_t[1]):int(v_t[3])] = 1
        # prev_tot_bt = np.sum(vp_mat)

        for c_t in c_tiles:
            cache_mat = np.zeros((10, 20))
            cache_mat[int(c_t[0]):int(c_t[2]), int(c_t[1]):int(c_t[3])] = 1

            # check intersection
            overlap = vp_mat + cache_mat

            vp_mat[overlap == 2] = 0

        # curr_tot_bt = np.sum(vp_mat)

        # if prev_tot_bt == curr_tot_bt:
        #     tiles_to_fetch.append(v_t)
        # else:
        #     mat_repartition += vp_mat
        mat_repartition += vp_mat

    mat_repartition[mat_repartition > 0] = 1

    if np.sum(mat_repartition)>0:

        # -------------- Part of VASTile implementation for furthe partitioning the tiles starts -------------- #
        # partition the rectangular polygons to perfect rectangles.
        new_tiles = repartition_given_polygon.repartition_tiles(mat_repartition, video_name, f, gamma, n)
        for t in new_tiles:
            tiles_to_fetch.append(t)
        # -------------- Part of VASTile implementation for furthe partitioning the tiles ends --------------- #

    stop_time = time.time()

    return tiles_to_fetch, stop_time - start_time
