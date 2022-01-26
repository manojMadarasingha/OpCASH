# further partition the new tiles that are streamed from the servers newly
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches
from skimage import measure
import os
import settings


# create store the indices
def store_new_t(new_t, data_store_path, n, u):
    columns = ['l_l_m', 'l_l_n', 'u_r_m', 'u_r_n', 'n', 'u']
    new_t = np.asarray(new_t)
    request_order = np.repeat([n], new_t.shape[0]).reshape([-1, 1])
    user_num = np.repeat([u], new_t.shape[0]).reshape([-1, 1])

    if len(new_t) > 0:
        new_arr = np.concatenate([new_t, request_order, user_num], axis=1)
        df = pd.DataFrame(columns=columns,
                          data=new_arr)
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(data_store_path + '/user_' + str(n) + '/tiles_new.csv', index=False)

    return


# given a tile, check whether it needs to be further partitioned or not
# in the horizontal or vertical direction.
# return a boolean value if to be partitioned.
def check_given_tile_to_be_partitioned(tile, gamma):
    # take the vertical tile position of the tile center.
    tile_center = (tile[2] - tile[0]) / 2 + tile[0]

    # tile angles are corresponding to the tile position. original tiles
    # convert it to nearaest even number for the ease of division
    # max_vertical_tiles = [3, 4, 3, 3, 5, 6]
    # max_hori_tiles = [14, 12, 12, 9, 7, 6]

    # alpha =1
    if gamma == 1:
        max_vertical_tiles = [4, 4, 4, 4, 6, 6]
        max_hori_tiles = [14, 12, 12, 10, 8, 6]
    elif gamma == 0.5:
        max_vertical_tiles = [2, 2, 2, 2, 4, 4]
        max_hori_tiles = [8, 6, 6, 6, 4, 4]
    elif gamma == 0.25:
        max_vertical_tiles = [2, 2, 2, 2, 2, 2]
        max_hori_tiles = [4, 4, 4, 4, 2, 2]

    # converted the above tile values to the nearest even number value
    # to be compatible with the further tile partitioning algorithm where the entire frame
    # is comparable highly sensitive.
    # accept_verti_tiles = [2, 4, 2, 2, 4, 4]
    # accept_hori_tiles = [4, 4, 4, 4, 2, 2]

    accept_verti_tiles = max_vertical_tiles
    accept_hori_tiles = max_hori_tiles

    # tile_center % 5 ==> directly gives the corresponding index of max vertical/horizontal tiles arrays
    # if the tile center is >=5, subtract from 9.5 (maximum possible vertical center) to get the value
    # between 0-4 which can be again used get the corresponding index position.
    if tile_center >= 5:
        tile_center = (9.5 - tile_center)

    correspond_tile_arr_index = int(tile_center % 5)

    # maximum allowable tiles for the partition
    max_hori_tile = max_hori_tiles[correspond_tile_arr_index]
    max_verti_tile = max_vertical_tiles[correspond_tile_arr_index]

    acceptable_hori_tile = accept_hori_tiles[correspond_tile_arr_index]
    accept_verti_tile = accept_verti_tiles[correspond_tile_arr_index]

    is_vertical_partition = False
    is_horizontal_partition = False

    if tile[2] - tile[0] >= max_verti_tile and tile[3] - tile[1] >= max_hori_tile:
        is_vertical_partition = True
        is_horizontal_partition = True
    elif tile[2] - tile[0] >= max_verti_tile and tile[3] - tile[1] < max_hori_tile:
        is_vertical_partition = True
        is_horizontal_partition = False
    elif tile[2] - tile[0] < max_verti_tile and tile[3] - tile[1] >= max_hori_tile:
        is_vertical_partition = False
        is_horizontal_partition = True

    return is_horizontal_partition, is_vertical_partition, acceptable_hori_tile, accept_verti_tile


# ===== horizontally algied tiles starts ===== #
# ===== horizontally algied tiles starts ===== #
# given the absolute tile coordinates get the
# horizontal alignment of the tile
def get_hori_tile_alignment(ori_tile):
    diff_fc_tb = np.abs(10 - ori_tile[1])
    diff_fc_tc = np.abs(10 - (ori_tile[1] + ori_tile[3]) // 2)
    diff_fc_te = np.abs(10 - ori_tile[3])

    diff_list = [diff_fc_te, diff_fc_tc, diff_fc_tb]

    return diff_list.index(min(diff_list))


def get_abs_tile_coord_right_aligned(ori_tile, max_hori_tiles):
    total_tliles = ori_tile[3] - ori_tile[1]
    num_of_max_tiles = int(total_tliles // max_hori_tiles)
    remaining_tiles = total_tliles % max_hori_tiles

    abs_tiles = []

    for t in range(num_of_max_tiles):
        abs_tiles.append(
            [ori_tile[0], ori_tile[1] + (t) * max_hori_tiles,
             ori_tile[2], ori_tile[1] + (t + 1) * max_hori_tiles])

    abs_tiles.append(
        [ori_tile[0], ori_tile[1] + (num_of_max_tiles) * max_hori_tiles,
         ori_tile[2], ori_tile[3]])

    new_abs_tiles = []
    for t in abs_tiles:
        if t[3] - t[1] > 0:
            new_abs_tiles.append(t)

    return new_abs_tiles


def get_abs_tile_coord_left_aligned(ori_tile, max_hori_tiles):
    total_tliles = ori_tile[3] - ori_tile[1]
    num_of_max_tiles = int(total_tliles // max_hori_tiles)
    remaining_tiles = total_tliles % max_hori_tiles

    abs_tiles = []

    for t in range(num_of_max_tiles, 0, -1):
        abs_tiles.append(
            [ori_tile[0], ori_tile[3] - (num_of_max_tiles - (t - 1)) * max_hori_tiles,
             ori_tile[2], ori_tile[3] - (num_of_max_tiles - (t)) * max_hori_tiles])

    abs_tiles.append(
        [ori_tile[0], ori_tile[1],
         ori_tile[2], ori_tile[3] - (num_of_max_tiles * max_hori_tiles)])

    new_abs_tiles = []
    for t in abs_tiles:
        if t[3] - t[1] > 0:
            new_abs_tiles.append(t)

    return new_abs_tiles


def get_abs_tile_coord_center_aligned_hori(ori_tile, max_hori_tiles):
    # apply the center tile
    # divide the main tile into two parts. apply left and right alignment
    t_c = (ori_tile[1] + ori_tile[3]) // 2
    left_tiles = [ori_tile[0], ori_tile[1], ori_tile[2], t_c - max_hori_tiles // 2]
    center_tile = [ori_tile[0], t_c - max_hori_tiles // 2, ori_tile[2], t_c + max_hori_tiles // 2]
    right_tiles = [ori_tile[0], t_c + max_hori_tiles // 2, ori_tile[2], ori_tile[3]]

    new_abs_tiles = []

    # add left tiles
    if left_tiles[3] > left_tiles[1]:
        for new_t in get_abs_tile_coord_left_aligned(left_tiles, max_hori_tiles):
            new_abs_tiles.append(new_t)

    # add center tile
    new_abs_tiles.append(center_tile)

    # add right tiles
    if right_tiles[3] > right_tiles[1]:
        for new_t in get_abs_tile_coord_right_aligned(right_tiles, max_hori_tiles):
            new_abs_tiles.append(new_t)

    return new_abs_tiles


# ===== horizontally algied tiles stops ===== #
# ===== horizontally algied tiles stops ===== #

# ===== vertically algied tiles starts ===== #
# ===== vertically algied tiles starts ===== #
# given the absolute tile coordinates get the
# horizontal alignment of the tile
def get_verti_tile_alignment(ori_tile):
    diff_fc_tb = np.abs(5 - ori_tile[0])
    diff_fc_tc = np.abs(5 - (ori_tile[0] + ori_tile[2]) // 2)
    diff_fc_te = np.abs(5 - ori_tile[2])

    diff_list = [diff_fc_te, diff_fc_tc, diff_fc_tb]

    return diff_list.index(min(diff_list))


def get_abs_tile_coord_lower_aligned(ori_tile, max_verti_tiles):
    total_tliles = ori_tile[2] - ori_tile[0]
    num_of_max_tiles = int(total_tliles // max_verti_tiles)
    remaining_tiles = total_tliles % max_verti_tiles

    abs_tiles = []

    for t in range(num_of_max_tiles):
        abs_tiles.append(
            [ori_tile[0] + (t) * max_verti_tiles, ori_tile[1],
             ori_tile[0] + (t + 1) * max_verti_tiles, ori_tile[3]])

    abs_tiles.append(
        [ori_tile[0] + (num_of_max_tiles) * max_verti_tiles, ori_tile[1],
         ori_tile[2], ori_tile[3]])

    new_abs_tiles = []
    for t in abs_tiles:
        if t[2] - t[0] > 0:
            new_abs_tiles.append(t)

    return new_abs_tiles


def get_abs_tile_coord_upper_aligned(ori_tile, max_verti_tiles):
    total_tliles = ori_tile[2] - ori_tile[0]
    num_of_max_tiles = int(total_tliles // max_verti_tiles)
    remaining_tiles = total_tliles % max_verti_tiles

    abs_tiles = []

    for t in range(num_of_max_tiles, 0, -1):
        abs_tiles.append(
            [ori_tile[2] - (num_of_max_tiles - (t - 1)) * max_verti_tiles, ori_tile[1],
             ori_tile[2] - (num_of_max_tiles - (t)) * max_verti_tiles, ori_tile[3]])

    abs_tiles.append(
        [ori_tile[0], ori_tile[1],
         ori_tile[2] - (num_of_max_tiles * max_verti_tiles), ori_tile[3]])

    new_abs_tiles = []
    for t in abs_tiles:
        if t[2] - t[0] > 0:
            new_abs_tiles.append(t)

    return new_abs_tiles


def get_abs_tile_coord_center_aligned_verti(ori_tile, max_verti_tiles):
    # apply the center tile
    # divide the main tile into two parts. apply upper and lower alignment
    t_c = (ori_tile[0] + ori_tile[2]) // 2
    upper_tiles = [ori_tile[0], ori_tile[1], t_c - max_verti_tiles // 2, ori_tile[3]]
    center_tile = [t_c - max_verti_tiles // 2, ori_tile[1], t_c + max_verti_tiles // 2, ori_tile[3]]
    lower_tiles = [t_c + max_verti_tiles // 2, ori_tile[1], ori_tile[2], ori_tile[3]]

    new_abs_tiles = []

    # add left tiles
    if upper_tiles[2] > upper_tiles[0]:
        for new_t in get_abs_tile_coord_upper_aligned(upper_tiles, max_verti_tiles):
            new_abs_tiles.append(new_t)

    # add center tile
    new_abs_tiles.append(center_tile)

    # add right tiles
    if lower_tiles[2] > lower_tiles[0]:
        for new_t in get_abs_tile_coord_lower_aligned(lower_tiles, max_verti_tiles):
            new_abs_tiles.append(new_t)

    return new_abs_tiles


def partition_new_tiles(new_t, data_store_path, n, u, gamma, ena_store):
    start_time = time.time()

    initial_tiles_before_further_partition = []
    initial_tiles_after_further_partition_all = []
    for t_ind, tile in enumerate(new_t):

        # tile = np.append(tile, [t_ind])
        initial_tiles_before_further_partition.append(tile)

        partition_info = check_given_tile_to_be_partitioned(tile, gamma)

        # horizontally partition
        if partition_info[0] and not partition_info[1]:

            alignment = get_hori_tile_alignment(tile)

            if alignment == settings.LEFT_ALIGNED:
                new_abs_tiles = get_abs_tile_coord_left_aligned(tile, partition_info[2])
            elif alignment == settings.CENTER_ALIGNED:
                new_abs_tiles = get_abs_tile_coord_center_aligned_hori(tile, partition_info[2])
            else:
                new_abs_tiles = get_abs_tile_coord_right_aligned(tile, partition_info[2])

            max_pertile_intensity = 0
            max_pertile_intensity_tile = []
            for new_tile in new_abs_tiles:
                # calculate avg saliency given the tile
                new_tile = np.asarray(new_tile).astype(int)
                # new_tile = np.append(new_tile, [t_ind])
                # per_tile_intensity = cal_pertile_intensity(new_tile, comb_viewport_map)
                # new_tile = np.concatenate([new_tile, np.asarray([per_tile_intensity, tile[-2], t_ind])])

                # if max_pertile_intensity < per_tile_intensity:
                #     max_pertile_intensity = per_tile_intensity
                #     max_pertile_intensity_tile = new_tile

                initial_tiles_after_further_partition_all.append(new_tile)
            # if len(max_pertile_intensity_tile) > 0:
            #     initial_tiles_after_further_partition_max.append(max_pertile_intensity_tile)


        # vertically partition
        elif not partition_info[0] and partition_info[1]:

            alignment = get_verti_tile_alignment(tile)

            if alignment == settings.UPPER_ALIGNED:
                new_abs_tiles = get_abs_tile_coord_upper_aligned(tile, partition_info[3])
            elif alignment == settings.CENTER_ALIGNED:
                new_abs_tiles = get_abs_tile_coord_center_aligned_verti(tile, partition_info[3])
            else:
                new_abs_tiles = get_abs_tile_coord_lower_aligned(tile, partition_info[3])

            max_pertile_intensity = 0
            max_pertile_intensity_tile = []
            for new_tile in new_abs_tiles:
                # calculate avg saliency given the tile
                new_tile = np.asarray(new_tile).astype(int)
                # new_tile = np.append(new_tile, [t_ind])
                # per_tile_intensity = cal_pertile_intensity(new_tile, comb_viewport_map)
                # new_tile = np.concatenate([new_tile, np.asarray([per_tile_intensity, tile[-2], t_ind])])

                # if max_pertile_intensity < per_tile_intensity:
                #     max_pertile_intensity = per_tile_intensity
                #     max_pertile_intensity_tile = new_tile

                initial_tiles_after_further_partition_all.append(new_tile)
            # if len(max_pertile_intensity_tile) > 0:
            #     initial_tiles_after_further_partition_max.append(max_pertile_intensity_tile)

        # if partition should be done in both direction
        elif partition_info[0] and partition_info[1]:

            alignment = get_hori_tile_alignment(tile)

            if alignment == settings.LEFT_ALIGNED:
                temp_hori_new_abs_tiles = get_abs_tile_coord_left_aligned(tile, partition_info[2])
            elif alignment == settings.CENTER_ALIGNED:
                temp_hori_new_abs_tiles = get_abs_tile_coord_center_aligned_hori(tile, partition_info[2])
            else:
                temp_hori_new_abs_tiles = get_abs_tile_coord_right_aligned(tile, partition_info[2])

            # for detected each of the tile do the partitioning in the vertical axis
            for new_t_hori in temp_hori_new_abs_tiles:

                # get the vertical alignment
                alignment = get_verti_tile_alignment(new_t_hori)

                # partition the tile vertically
                if alignment == settings.UPPER_ALIGNED:
                    new_abs_tiles = get_abs_tile_coord_upper_aligned(new_t_hori, partition_info[3])
                elif alignment == settings.CENTER_ALIGNED:
                    new_abs_tiles = get_abs_tile_coord_center_aligned_verti(new_t_hori,
                                                                            partition_info[3])
                else:
                    new_abs_tiles = get_abs_tile_coord_lower_aligned(new_t_hori, partition_info[3])

                for new_tile in new_abs_tiles:
                    # calculate avg saliency given the tile
                    new_tile = np.asarray(new_tile).astype(int)
                    # new_tile = np.append(new_tile, [t_ind])
                    # per_tile_intensity = cal_pertile_intensity(new_tile, comb_viewport_map)
                    # new_tile = np.concatenate([new_tile, np.asarray([per_tile_intensity, tile[-2], t_ind])])
                    initial_tiles_after_further_partition_all.append(new_tile)

        else:
            initial_tiles_after_further_partition_all.append(tile)

    stop_time = time.time()
    if ena_store:
        store_new_t(initial_tiles_after_further_partition_all, data_store_path, n, u)

    return initial_tiles_after_further_partition_all, stop_time - start_time
