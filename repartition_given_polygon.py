

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

# import pre_processing_steps
# import optimized_tile_partitioning
# import find_rectangulars
# import supportive_code_snipptes
from skimage import measure

import settings
from Tile_repartitioning import find_rectangulars, \
    pre_processing_steps, optimized_tile_partitioning, \
    supportive_code_snipptes


# ==============start simple pre-procesing of the saliency maps ==============
# ==============start simple pre-procesing of the saliency maps ==============
# down sample the image from the
def down_sample_image(im):
    down_sam_image_linear_right_rotated = im.transpose()
    return im, down_sam_image_linear_right_rotated


# tiling get the score from the histogram equalized image
# tiling get the score from the histogram equalized image
def tile_frame_and_get_sal_score(sal_img, hori_tiles, verti_tiles):
    c = verti_tiles
    r = hori_tiles

    im_h = sal_img.shape[0]
    im_w = sal_img.shape[1]

    thresholded_img_approxi = np.where(sal_img > 0, 1, 0)

    # remove all the blobs keeping only a largest blob in the thoresholded image frame.
    blobs_labels = measure.label(thresholded_img_approxi, background=0, connectivity=1)
    unique, count = np.unique(blobs_labels, return_counts=True)
    unique = list(unique)
    count = list(count)

    unique = unique[1:]
    count = count[1:]

    count = np.asarray(count)
    unique = np.asarray(unique)

    count_sorted = count.argsort()
    unique = np.flip(unique[count_sorted])

    all_score_mat_approxi = []

    for b_id in unique:
        score_mat_approxi = np.zeros((r, c))
        score_mat_approxi[blobs_labels == b_id]=1
        all_score_mat_approxi.append(score_mat_approxi)

    return all_score_mat_approxi


# get the center of the given cluster.
# specially to validate the small cluster given
def centeroidnp(arr, im_w, tile_w):
    length = arr.shape[0]
    # sum_h = np.sum(arr[:, 0])
    sum_w = np.sum(arr[:, 1])

    # center_h = sum_h/length
    center_w = sum_w / length

    if (center_w < 2 * tile_w) or ((im_w - 2 * tile_w) < center_w):
        return False
    else:
        return True


# ==============start simple pre-procesing of the saliency maps ends==============
# ==============start simple pre-procesing of the saliency maps ends==============


# detect the vertices of the binary polygon
# convex = 1
# colinear = 2
# concave = -1

def detect_vertices(img):
    w = img.shape[1]
    h = img.shape[0]

    # img[14, 4] = 1

    verti_mat = np.zeros((h + 1, w + 1))

    hori_detector = np.asarray([-1, 1]).reshape([2, 1])
    verti_detector = np.asarray([-1, 1]).reshape([1, 2])

    # traverse through all the points.
    # identify the tiles correspond to that particuar tile
    for i in range(h + 1):
        for j in range(w + 1):

            # 1/0 tile ind in horizontal direction
            h_l = j - 1
            h_r = j

            v_u = i - 1
            v_d = i

            # define the tile coordinates in the 1/0 map
            if i == 0:
                if j == 0:
                    coord = [[v_d, h_r]]
                    if img[coord[0][0], coord[0][1]] == 1:
                        verti_mat[i, j] = 1
                elif j == w:
                    coord = [[v_d, h_l]]
                    if img[coord[0][0], coord[0][1]] == 1:
                        verti_mat[i, j] = 1
                else:
                    coord = [[v_d, h_l], [v_d, h_r]]
                    tot = 0
                    for c in coord:
                        tot += img[c[0], c[1]]
                    if tot == 1:
                        verti_mat[i, j] = 1

            elif i == h:
                if j == 0:
                    coord = [[v_u, h_r]]
                    if img[coord[0][0], coord[0][1]] == 1:
                        verti_mat[i, j] = 1
                elif j == w:
                    coord = [[v_u, h_l]]
                    if img[coord[0][0], coord[0][1]] == 1:
                        verti_mat[i, j] = 1
                else:
                    coord = [[v_u, h_l], [v_u, h_r]]
                    tot = 0
                    for c in coord:
                        tot += img[c[0], c[1]]
                    if tot == 1:
                        verti_mat[i, j] = 1

            else:
                if j == 0:
                    coord = [[v_u, h_r], [v_d, h_r]]
                    tot = 0
                    for c in coord:
                        tot += img[c[0], c[1]]
                    if tot == 1:
                        verti_mat[i, j] = 1
                elif j == w:
                    coord = [[v_u, h_l], [v_d, h_l]]
                    tot = 0
                    for c in coord:
                        tot += img[c[0], c[1]]
                    if tot == 1:
                        verti_mat[i, j] = 1
                else:
                    coord = [[v_u, h_l], [v_u, h_r], [v_d, h_l], [v_d, h_r]]
                    tot = 0
                    for c in coord:
                        tot += img[c[0], c[1]]
                    if tot == 1:
                        verti_mat[i, j] = 1
                    elif tot == 3:
                        verti_mat[i, j] = -1

            # identify the vertices with

    # get the boundary line
    # edges = cv2.Canny(img)
    # plt.imshow(verti_mat, cmap='hot', interpolation='none')
    # plt.show()
    return verti_mat


# detect the colinear vertices given the convex and the concave vertices
def detect_colinear_vertices(convex_concave_mat):
    w = convex_concave_mat.shape[1]
    h = convex_concave_mat.shape[0]

    # find the left upper most coordinate
    left_upper_convex = []
    for i in range(convex_concave_mat.shape[0]):
        for j in range(convex_concave_mat.shape[1]):
            if convex_concave_mat[i, j] == 1:
                left_upper_convex.extend([i, j, 1])
                break
        if len(left_upper_convex) > 0:
            break

    # find all the colinear vertices traversing through clockwise direction starting
    # starting from the upper left coordinate

    # count all vertices in the convex_concav_mat
    num_of_vertices = len(np.argwhere(convex_concave_mat != 0))

    current_verti = left_upper_convex
    prev_coord = []
    traverse_direction = 0
    # 1:downward
    # 2:upward
    # 3:rightward
    # 4:leftward

    for v in range(num_of_vertices):

        # since this is the left upper most coordinate directly shoot to the right direction
        if v == 0:
            i = current_verti[0]
            for j in range(current_verti[1] + 1, w):

                if convex_concave_mat[i, j] == 0:
                    convex_concave_mat[i, j] = 2
                else:
                    prev_coord = current_verti
                    current_verti = [i, j, convex_concave_mat[i, j]]
                    traverse_direction = 3
                    break

        else:
            i = current_verti[0]
            j = current_verti[1]

            if current_verti[2] == 1:
                if traverse_direction == 1:  # if the current coordinate = 1 previous coordinate is convex =1
                    traverse_direction = 4
                    const_coord = 'i'
                    const_coord_ind = i
                    var_coord = j - 1
                    limit = -1
                    step = -1

                elif traverse_direction == 2:
                    traverse_direction = 3
                    const_coord = 'i'
                    const_coord_ind = i
                    var_coord = j + 1
                    limit = w
                    step = 1

                elif traverse_direction == 3:
                    traverse_direction = 1
                    const_coord = 'j'
                    const_coord_ind = j
                    var_coord = i + 1
                    limit = h
                    step = 1

                else:
                    traverse_direction = 2
                    const_coord = 'j'
                    const_coord_ind = j
                    var_coord = i - 1
                    limit = -1
                    step = -1

            else:
                if traverse_direction == 1:
                    traverse_direction = 3
                    const_coord = 'i'
                    const_coord_ind = i
                    var_coord = j + 1
                    limit = w
                    step = 1

                elif traverse_direction == 2:
                    traverse_direction = 4
                    const_coord = 'i'
                    const_coord_ind = i
                    var_coord = j - 1
                    limit = -1
                    step = -1

                elif traverse_direction == 3:
                    traverse_direction = 2
                    const_coord = 'j'
                    const_coord_ind = j
                    var_coord = i - 1
                    limit = -1
                    step = -1

                else:
                    traverse_direction = 1
                    const_coord = 'j'
                    const_coord_ind = j
                    var_coord = i + 1
                    limit = h
                    step = 1

            for v in range(var_coord, limit, step):

                if const_coord == 'j':
                    if convex_concave_mat[v, const_coord_ind] == 0:
                        convex_concave_mat[v, const_coord_ind] = 2
                    else:
                        current_verti = [v, const_coord_ind, convex_concave_mat[v, const_coord_ind]]
                        break

                else:
                    if convex_concave_mat[const_coord_ind, v] == 0:
                        convex_concave_mat[const_coord_ind, v] = 2
                    else:
                        current_verti = [const_coord_ind, v, convex_concave_mat[const_coord_ind, v]]
                        break

    # plt.imshow(convex_concave_mat)
    # plt.show()

    return convex_concave_mat


# traverse through the vertex path. start from the left upper most valid positiona and travese
# in clock wise direction to identify the vertices and their neighbours. This is specially to figure out neighbour
# coordinates.
def get_neighbourhood_indices(verti_mat, total_coord):
    w = verti_mat.shape[1]
    h = verti_mat.shape[0]

    # traverse through the vertices and identify neighbourhoud pairs
    # After identifying the first convex vertex, traverse through the array finding the neighbourhood indices

    # contains all the indices related data
    indices_chart = []

    # identify the first upper left most convex element
    curr_m = 0
    curr_n = 0
    vert_found = False
    for i in range(h):
        for j in range(w):
            if verti_mat[i, j] == 1:
                # if the found vertex is at the righter most border of the frame, we cannot traverse to the right direction
                # therefore omit this cooridinate
                if j == w - 1:
                    continue
                curr_m = i
                curr_n = j
                vert_found = True
                break
        if vert_found:
            break

    # traverse through the vertices matrix identifying the neighbourhood vertices
    # print(1)
    init_m = curr_m
    init_n = curr_n

    next_m = -1
    next_n = -1

    # increment factor for traversing
    m = 0
    n = 1
    count = 0
    coord_relat_list = []

    error_count = 0
    error_count_limit = 10
    frame_condition = settings.FRAME_OK

    while not ((next_m == init_m) and (
            next_n == init_n)):  # (count < total_coord):  # not ((next_m == init_m) and (next_n == init_n))
        count += 1

        # if count == 27:
        #     print(1)

        temp_coord_list = []
        temp_coord_list.append([curr_m, curr_n])

        if n != 0:
            if n == 1:
                limit = w
                start = curr_n + 1
            else:
                limit = -1
                start = curr_n - 1

            for j in range(start, limit, n):
                # found convex
                if verti_mat[curr_m, j] == 1:
                    next_m = curr_m
                    next_n = j
                    temp_coord_list.append([next_m, next_n])
                    if n == 1:
                        m = 1
                    else:
                        m = -1
                    n = 0
                    break

                # found concave
                elif verti_mat[curr_m, j] == -1:
                    next_m = curr_m
                    next_n = j
                    temp_coord_list.append([next_m, next_n])
                    if n == 1:
                        m = -1
                    else:
                        m = 1
                    n = 0
                    break
        elif m != 0:
            if m == 1:
                limit = h
                start = curr_m + 1
            else:
                limit = -1
                start = curr_m - 1

            for i in range(start, limit, m):
                # found convex
                if verti_mat[i, curr_n] == 1:
                    next_m = i
                    next_n = curr_n
                    temp_coord_list.append([next_m, next_n])
                    if m == 1:
                        n = -1
                    else:
                        n = 1
                    m = 0
                    break

                # found concave
                elif verti_mat[i, curr_n] == -1:
                    next_m = i
                    next_n = curr_n
                    temp_coord_list.append([next_m, next_n])
                    if m == 1:
                        n = 1
                    else:
                        n = -1
                    m = 0
                    break
        if len(temp_coord_list) == 1:
            print(1)
            error_count += 1

        if error_count > error_count_limit:
            frame_condition = settings.FRAME_ERROR
            break

        curr_m = next_m
        curr_n = next_n
        coord_relat_list.append(temp_coord_list)

    return coord_relat_list, frame_condition


# partition the rectilinear polygon
def detect_chords(verti_mat, neighbour_lists, convex_ind, colinear_ind, concave_inds, holes):
    # all the concave points of the polygon
    concave_inds = np.argwhere(verti_mat == -1)

    vertical_chords = []
    horizontal_chords = []
    # find vertical coords
    for i in range(len(concave_inds)):
        for j in range(i + 1, len(concave_inds)):
            if concave_inds[i][1] == concave_inds[j][1]:
                first_coord = list(concave_inds[i])
                second_coord = list(concave_inds[j])
                # check whether obtained coordinates are consecutive
                valid_cord = True
                for v_pair in neighbour_lists:
                    if (first_coord in v_pair) and (second_coord in v_pair):
                        valid_cord = False
                    # check if any concave cooridnate between the given 2 coordinate. If so, remove that coordinate
                    # check also for if there are any convex/colinear coordinates as well
                    if first_coord[0] < second_coord[0]:
                        min_m = first_coord[0]
                        max_m = second_coord[0]
                    else:
                        max_m = first_coord[0]
                        min_m = second_coord[0]
                    common_n = first_coord[1]

                    for c in concave_inds:
                        if c[1] == common_n:
                            if c[0] > min_m and c[0] < max_m:
                                valid_cord = False
                                break
                    if valid_cord:
                        for c in convex_ind:
                            if c[1] == common_n:
                                if c[0] > min_m and c[0] < max_m:
                                    valid_cord = False
                                    break
                    if valid_cord:
                        for c in colinear_ind:
                            if c[1] == common_n:
                                if c[0] > min_m and c[0] < max_m:
                                    valid_cord = False
                                    break

                if valid_cord:
                    vertical_chords.append([first_coord, second_coord])

            if concave_inds[i][0] == concave_inds[j][0]:
                first_coord = list(concave_inds[i])
                second_coord = list(concave_inds[j])
                # check whether obtained coordinates are consecutive
                valid_cord = True
                for v_pair in neighbour_lists:
                    if (first_coord in v_pair) and (second_coord in v_pair):
                        valid_cord = False
                    # check if any concave cooridnate between the selected 2 coordinate. If so, remove that coordinate
                    # check for any convex or colinear vertices
                    if first_coord[1] < second_coord[1]:
                        min_n = first_coord[1]
                        max_n = second_coord[1]
                    else:
                        max_n = first_coord[1]
                        min_n = second_coord[1]
                    common_m = first_coord[0]

                    for c in concave_inds:
                        if c[0] == common_m:
                            if c[1] > min_n and c[1] < max_n:
                                valid_cord = False
                                break

                    if valid_cord:
                        for c in convex_ind:
                            if c[0] == common_m:
                                if c[1] > min_n and c[1] < max_n:
                                    valid_cord = False
                                    break
                    if valid_cord:
                        for c in colinear_ind:
                            if c[0] == common_m:
                                if c[1] > min_n and c[1] < max_n:
                                    valid_cord = False
                                    break

                    if not valid_cord:
                        break

                if valid_cord:
                    horizontal_chords.append([first_coord, second_coord])

    # if holes are present
    if holes != None:
        horizontal_chords, vertical_chords = supportive_code_snipptes.find_chords_combined_with_holes(holes,
                                                                                                      concave_inds,
                                                                                                      horizontal_chords,
                                                                                                      vertical_chords,
                                                                                                      neighbour_lists)

    return horizontal_chords, vertical_chords


# get coordinates of the different types of vertices
def get_coordinates(i_j_arrays):
    coord_list = []
    for i in range(len(i_j_arrays[0])):
        coord_list.append([i_j_arrays[0][i], i_j_arrays[1][i]])

    return coord_list


# generate the complete boundary coordinate list.
def get_boundary_path(neighbour_lists):
    complete_path = []

    for vert1, vert2 in neighbour_lists:
        # if vert1 == [5, 4] and vert2 == [2, 4]:
        #     print(1)
        complete_path.append(vert1)
        # if the corrdinates are horizontally colinear
        if vert1[0] == vert2[0]:
            diff = vert2[1] - vert1[1]
            if diff < 0:
                start = vert1[1] - 1
                step = -1
            else:
                start = vert1[1] + 1
                step = 1
            end = vert1[1] + diff

            for x in range(start, end, step):
                complete_path.append([vert1[0], x])

        # if the corrdinates are vertically colinear
        else:
            diff = vert2[0] - vert1[0]
            if diff < 0:
                start = vert1[0] - 1
                step = -1
            else:
                start = vert1[0] + 1
                step = 1
            end = vert1[0] + diff

            for x in range(start, end, step):
                complete_path.append([x, vert1[1]])

    return complete_path


def fit_given_poly_partition_to_2d_array(ploygon, w, h):
    arr = np.zeros((h, w))
    for cell in ploygon:
        arr[cell[0], cell[1]] = 1

    return arr


# calculate the generalized saliency
def calculate_saliency(rect, sal_frame, hori_tiles, verti_tiles):
    # x is vertical direction
    # y is horizonatal direction
    tile_h = sal_frame.shape[0] // verti_tiles
    tile_w = sal_frame.shape[1] // hori_tiles

    upper_left_x = int(tile_h * rect[0][0])
    upper_left_y = int(tile_w * rect[0][1])
    lower_right_x = int(tile_h * rect[2][0])
    lower_right_y = int(tile_w * rect[2][1])

    tot_sal = sal_frame[upper_left_x:lower_right_x, upper_left_y:lower_right_y].sum()
    avg_sal = tot_sal / (np.abs(rect[2][1] - rect[0][1]) * np.abs(rect[2][0] - rect[0][0]))

    return tot_sal, avg_sal


def repartition_tiles(tile_mat, video_name, f, gamma,n):
    hori_tiles = 20
    verti_tiles = 10


    im, down_sam_image_linear_right_rotated = down_sample_image(tile_mat)


    score_mat_arr = tile_frame_and_get_sal_score(down_sam_image_linear_right_rotated, hori_tiles, verti_tiles)


    all_filtered_fov_rectangles = []

    # run the process for each blob detected iteratively
    # to avoid any potential finer blobs missmatching consider only the blobs
    # having sufficient finer thresholding value
    overall_frame_cond = settings.FRAME_OK
    for b in range(len(score_mat_arr)):
        score_mat = score_mat_arr[b]

        hole_removed_img = pre_processing_steps.remove_holes(
            score_mat)
        # old verion: thresholding based on the 20 x 10 tiles 'processed_score_mat' was the argument
        # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        # plt.imshow(hole_removed_img)
        # plt.show()
        # plt.close()

        # Identify the most highlighted slaiency regions in the reigon
        salient_region = np.zeros(hole_removed_img.shape)
        high_middle_zero_salient_image = pre_processing_steps.combine_apprpxim_img_with_finer_image(
            hole_removed_img, salient_region)

        # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        # plt.imshow(high_middle_zero_salient_image)
        # plt.show()
        # plt.close()

        # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        # plt.imshow(high_middle_zero_salient_image)
        # plt.show()

        # remove most salient regions and identify single perfect rectangle areas and feed the other rectlinear polygon regions for further process
        perfect_rectangles_middle_sal, rect_polygon_without_holes_middle_sal, \
        rect_polygon_with_holes_middle_sale, perfect_rectangles_high_sal, \
        rect_polygon_without_holes_high_sal, count_perfect_holes, count_imperfect_holes = pre_processing_steps.identfiy_perfect_rectangles_and_rectilinear_polygons(
            high_middle_zero_salient_image)

        # add perfect rectangeles
        for rect in perfect_rectangles_middle_sal:
            all_filtered_fov_rectangles.append(rect)

        for rect in perfect_rectangles_high_sal:
            all_filtered_fov_rectangles.append(rect)

        # for rect in perfect_rects_oov:
        #     all_filtered_oov_rectangles.append(rect)

        all_neighbour_list_chords = []
        all_nearest_chords = []
        all_independent_chords = []

        # =============== getting required rectangular partitions starts ====================
        # +++++++treat the rect polygons without holes for high salient regions++++++++++++++.
        for polygon_wihtout_holes in rect_polygon_without_holes_high_sal:
            # fit the rectlinear polygon to w x h frame
            score_mat_ref = np.zeros((hori_tiles, verti_tiles))

            for i in polygon_wihtout_holes:
                score_mat_ref[i[0], i[1]] = 2

            # plt.imshow(score_mat_ref, cmap='hot', interpolation='nearest')
            # plt.show()
            polygon_fitted_array = fit_given_poly_partition_to_2d_array(polygon_wihtout_holes,
                                                                        high_middle_zero_salient_image.shape[
                                                                            1],
                                                                        high_middle_zero_salient_image.shape[
                                                                            0])

            convex_concave_mat = detect_vertices(polygon_fitted_array)
            convex_concave_colinear_mat = detect_colinear_vertices(convex_concave_mat)

            print('Get_vertices')
            convex_coord = get_coordinates(np.where(convex_concave_colinear_mat == 1))
            concave_coord = get_coordinates(np.where(convex_concave_colinear_mat == -1))
            colinear_coord = get_coordinates(np.where(convex_concave_colinear_mat == 2))

            print('Neighbour list')
            neighbour_lists, frame_cond = get_neighbourhood_indices(convex_concave_colinear_mat,
                                                                    len(convex_coord) + len(concave_coord))
            if frame_cond == settings.FRAME_ERROR:
                overall_frame_cond = frame_cond
                break

            complete_boundary_path = get_boundary_path(neighbour_lists)

            print('get chords')
            horizontal_cords, vertical_coord = detect_chords(convex_concave_colinear_mat, neighbour_lists,
                                                             convex_coord, colinear_coord, concave_coord,
                                                             holes=None)

            print('Partioning')
            independen_chord, nearest_chord = optimized_tile_partitioning.try_nx_for_getting_results(
                horizontal_chords=horizontal_cords,
                vertical_chords=vertical_coord,
                concave_vertices=concave_coord,
                convex_vertics=convex_coord,
                colinear_vertices=colinear_coord,
                boundary=convex_concave_colinear_mat,
                neighbour_list=complete_boundary_path,
                is_holes_contains=False,
                holes=None,
                video_name=video_name,
                frame_num=f,
                threshold=0.5,  # A dump value assigned as the this value is not used in this analysi
                is_fov=True,
                is_buffer=False
            )

            for chord in nearest_chord:
                all_nearest_chords.append(chord)
            for chord in independen_chord:
                all_independent_chords.append(chord)
            for chord in neighbour_lists:
                all_neighbour_list_chords.append(chord)

            # post_processing_of_image.find_rectangles(neighbour_lists, independen_chord,nearest_chord)
            all_rectangles = find_rectangulars.find_coord(neighbour_lists, concave_coord, convex_coord,
                                                          colinear_coord,
                                                          independen_chord,
                                                          nearest_chord,
                                                          is_holes_contains=False,
                                                          holes=None
                                                          )

            # fitler for removing overlapping rectangles
            filtered_rectangles = find_rectangulars.remove_overlapping_rectangles(all_rectangles)

            # add perfect rectangeles from polygon without holes
            for rect in filtered_rectangles:
                all_filtered_fov_rectangles.append(rect)

        if overall_frame_cond == settings.FRAME_ERROR:
            break

        # +++++++treat the rect polygons without holes for middle salient regions+++++++++++.
        for polygon_wihtout_holes in rect_polygon_without_holes_middle_sal:
            # fit the rectlinear polygon to w x h frame
            polygon_fitted_array = fit_given_poly_partition_to_2d_array(polygon_wihtout_holes,
                                                                        high_middle_zero_salient_image.shape[
                                                                            1],
                                                                        high_middle_zero_salient_image.shape[
                                                                            0])
            # if len(polygon_wihtout_holes) > 10 and f_ind == 35:
            #     print(1)
            #     polygon_fitted_array[13,4]=0

            # fill the polygon if it is empty
            convex_concave_mat = detect_vertices(polygon_fitted_array)
            convex_concave_colinear_mat = detect_colinear_vertices(convex_concave_mat)

            print('Get_vertices')
            convex_coord = get_coordinates(np.where(convex_concave_colinear_mat == 1))
            concave_coord = get_coordinates(np.where(convex_concave_colinear_mat == -1))
            colinear_coord = get_coordinates(np.where(convex_concave_colinear_mat == 2))

            print('Neighbour list')
            neighbour_lists, frame_cond = get_neighbourhood_indices(convex_concave_colinear_mat,
                                                                    len(convex_coord) + len(concave_coord))
            if frame_cond == settings.FRAME_ERROR:
                overall_frame_cond = frame_cond
                break

            complete_boundary_path = get_boundary_path(neighbour_lists)

            print('get chords')
            horizontal_cords, vertical_coord = detect_chords(convex_concave_colinear_mat, neighbour_lists,
                                                             convex_coord, colinear_coord, concave_coord,
                                                             holes=None)

            print('Partioning')
            independen_chord, nearest_chord = optimized_tile_partitioning.try_nx_for_getting_results(
                horizontal_chords=horizontal_cords,
                vertical_chords=vertical_coord,
                concave_vertices=concave_coord,
                convex_vertics=convex_coord,
                colinear_vertices=colinear_coord,
                boundary=convex_concave_colinear_mat,
                neighbour_list=complete_boundary_path,
                is_holes_contains=False,
                holes=None,
                video_name=video_name,
                frame_num=f,
                threshold=0.5,
                is_fov=True,
                is_buffer=False
            )

            for chord in nearest_chord:
                all_nearest_chords.append(chord)
            for chord in independen_chord:
                all_independent_chords.append(chord)
            for chord in neighbour_lists:
                all_neighbour_list_chords.append(chord)

            # post_processing_of_image.find_rectangles(neighbour_lists, independen_chord,nearest_chord)
            all_rectangles = find_rectangulars.find_coord(neighbour_lists, concave_coord, convex_coord,
                                                          colinear_coord,
                                                          independen_chord,
                                                          nearest_chord,
                                                          is_holes_contains=False,
                                                          holes=None
                                                          )

            # fitler for removing overlapping rectangles
            filtered_rectangles = find_rectangulars.remove_overlapping_rectangles(all_rectangles)

            # add perfect rectangeles from polygon without holes
            for rect in filtered_rectangles:
                all_filtered_fov_rectangles.append(rect)

        if overall_frame_cond == settings.FRAME_ERROR:
            break

        # treat the polygons with holes middle salient regions
        for polygon_with_holes in (rect_polygon_with_holes_middle_sale):
            # fit the rectlinear polygon to w x h frame
            polygon_fitted_array = fit_given_poly_partition_to_2d_array(polygon_with_holes[0],
                                                                        high_middle_zero_salient_image.shape[
                                                                            1],
                                                                        high_middle_zero_salient_image.shape[
                                                                            0])

            # fill the holes
            hole_removed_img = pre_processing_steps.remove_holes(polygon_fitted_array)

            # remove the holes of the rectilinear polygon. holes are the highest saliency regios
            convex_concave_mat = detect_vertices(hole_removed_img)
            convex_concave_colinear_mat = detect_colinear_vertices(convex_concave_mat)

            print('Get_vertices')
            convex_coord = get_coordinates(np.where(convex_concave_colinear_mat == 1))
            concave_coord = get_coordinates(np.where(convex_concave_colinear_mat == -1))
            colinear_coord = get_coordinates(np.where(convex_concave_colinear_mat == 2))

            print('Neighbour list')
            neighbour_lists, frame_cond = get_neighbourhood_indices(convex_concave_colinear_mat,
                                                                    len(convex_coord) + len(concave_coord))
            if frame_cond == settings.FRAME_ERROR:
                overall_frame_cond = frame_cond
                break

            complete_boundary_path = get_boundary_path(neighbour_lists)

            print('get chords')
            horizontal_cords, vertical_coord = detect_chords(convex_concave_colinear_mat, neighbour_lists,
                                                             convex_coord, colinear_coord, concave_coord,
                                                             holes=polygon_with_holes[1])

            independen_chord, nearest_chord = optimized_tile_partitioning.try_nx_for_getting_results(
                horizontal_chords=horizontal_cords,
                vertical_chords=vertical_coord,
                concave_vertices=concave_coord,
                convex_vertics=convex_coord,
                colinear_vertices=colinear_coord,
                boundary=convex_concave_colinear_mat,
                neighbour_list=complete_boundary_path,
                is_holes_contains=True,
                holes=polygon_with_holes[1],
                video_name=video_name,
                frame_num=f,
                threshold=0.5,
                is_fov=True,
                is_buffer=False
            )

            for chord in nearest_chord:
                all_nearest_chords.append(chord)
            for chord in independen_chord:
                all_independent_chords.append(chord)
            for chord in neighbour_lists:
                all_neighbour_list_chords.append(chord)

            # post_processing_of_image.find_rectangles(neighbour_lists, independen_chord,nearest_chord)
            all_rectangles = find_rectangulars.find_coord(neighbour_lists, concave_coord, convex_coord,
                                                          colinear_coord,
                                                          independen_chord,
                                                          nearest_chord,
                                                          is_holes_contains=True,
                                                          holes=polygon_with_holes[1]
                                                          )

            # fitler for removing overlapping rectangles
            filtered_rectangles = find_rectangulars.remove_overlapping_rectangles(all_rectangles)

            # add perfect rectangeles from polygon with holes
            for rect in filtered_rectangles:
                all_filtered_fov_rectangles.append(rect)

        if overall_frame_cond == settings.FRAME_ERROR:
            break

    # commneted start
    if overall_frame_cond == settings.FRAME_ERROR:
        log_path = '/Users/ckat9988/Documents/Research/MEC_assisted_streaming/' \
                   'data_and_results/ILP_solutions/cost_iou_dist_e2u_s2e' \
                   '/partition_solution/gamma_' + str(gamma)
        f = open(log_path + '/' + 'error_frames.txt', "a+")

        string = 'video: ' + video_name + ' frame: ' + str(f) + ' threshold: ' + str(
            0.0) + '\n'

        f.write(string)
        f.flush()
        f.close()
        return

    # time upto partitinoing

    all_rectangles_fov_rotated = []
    for rectangle in all_filtered_fov_rectangles:
        new_rectangle = []
        # for each coordinate of the rectangle
        for coord in rectangle:
            x = -coord[0]
            y = coord[1]

            X = np.round(x * np.cos(-np.pi / 2) - y * np.sin(-np.pi / 2))
            Y = np.round(x * np.sin(-np.pi / 2) + y * np.cos(-np.pi / 2))

            new_rectangle.append([X, Y])

        all_rectangles_fov_rotated.append(new_rectangle)

    all_rect_rows = []
    combined_list = all_rectangles_fov_rotated  # + all_filtered_oov
    for rect_id, rectnagle in enumerate(combined_list):
        single_rect_row = []
        for coord in rectnagle:
            single_rect_row.append(coord[0])
            single_rect_row.append(coord[1])
        all_rect_rows.append([single_rect_row[0],single_rect_row[1],single_rect_row[4],single_rect_row[5]])


    return all_rect_rows
