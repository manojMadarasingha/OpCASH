import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure


# function: important
# remove the holes in the selected 1/0 binary image of the tiles
def remove_holes(processed_score_mat):
    processed_score_mat = processed_score_mat.astype(np.uint8)
    im_floodfill_1 = processed_score_mat.copy()
    im_floodfill_2 = processed_score_mat.copy()
    im_floodfill_3 = processed_score_mat.copy()
    im_floodfill_4 = processed_score_mat.copy()

    # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    # plt.imshow(processed_score_mat)
    # plt.show()
    # plt.close()

    # im_floodfill[0,1]=0
    # im_floodfill[19, 1] = 0

    h, w = processed_score_mat.shape[:2]
    mask_1 = np.zeros((h + 2, w + 2), np.uint8)
    mask_2 = np.zeros((h + 2, w + 2), np.uint8)
    mask_3 = np.zeros((h + 2, w + 2), np.uint8)
    mask_4 = np.zeros((h + 2, w + 2), np.uint8)

    if im_floodfill_1[0, 0] == 0:
        cv2.floodFill(im_floodfill_1, mask_1, seedPoint=(0, 0), newVal=255, flags=4)  # upDiff=100
    if im_floodfill_2[0, w - 1] == 0:
        cv2.floodFill(im_floodfill_2, mask_2, seedPoint=(w - 1, 0), newVal=255, flags=4)  # upDiff=100
    if im_floodfill_3[h - 1, 0] == 0:
        cv2.floodFill(im_floodfill_3, mask_3, seedPoint=(0, h - 1), newVal=255, flags=4)  # upDiff=100
    if im_floodfill_4[h - 1, w - 1] == 0:
        cv2.floodFill(im_floodfill_4, mask_4, seedPoint=(w - 1, h - 1), newVal=255, flags=4)  # upDiff=100

    im_floodfill = im_floodfill_1 + im_floodfill_2 + im_floodfill_3 + im_floodfill_4

    # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    # plt.imshow(im_floodfill)
    # plt.show()
    # plt.close()

    im_floodfill[im_floodfill > 0] = 255
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    # plt.imshow(im_floodfill_inv)
    # plt.show()
    # plt.close()

    im_out = processed_score_mat | im_floodfill_inv

    # fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    # plt.imshow(im_out)
    # plt.show()
    # plt.close()

    im_out[im_out == 255] = 1
    im_out = im_out.astype(float)

    return im_out


# function: important
# combine the processed (hole removed and only one cluster detected) image with the 2/1/0 finer image
# expand the high salient region
def combine_apprpxim_img_with_finer_image(hole_removed_img, score_mat_finer):
    processed_image = hole_removed_img.copy()

    combined_img = hole_removed_img + score_mat_finer
    finer_thresh = 2

    ind_high = np.argwhere(combined_img > finer_thresh)

    # all the high salient regions are contained in the middle (approximate) salient region.
    # processed image: original high salient + processed middle salient + 0 salient
    for i in ind_high:
        processed_image[i[0], i[1]] = 2

    # extract only the high salient regions
    processed_iamge_high_sal_ind = np.argwhere(processed_image > 1)

    temp_processed_image_high_sal_blobs = np.zeros((processed_image.shape[0], processed_image.shape[1]))
    for i in processed_iamge_high_sal_ind:
        temp_processed_image_high_sal_blobs[i[0], i[1]] = 1

    # identify each contours with highest saliency
    all_labels = measure.label(temp_processed_image_high_sal_blobs)
    blobs_labels = measure.label(all_labels, background=0, neighbors=4)

    unique, count = np.unique(blobs_labels, return_counts=True)

    unique = list(unique)
    count = list(count)

    # omitting the background blob, identify the boundaries of each contour
    expanded_high_sal_blobs = np.zeros((processed_image.shape[0], processed_image.shape[1]))
    for i in range(1, len(unique)):
        temp_blob = np.argwhere(blobs_labels == unique[i])
        r = temp_blob[:, 0]
        c = temp_blob[:, 1]

        for r_ind in range(r.min(), r.max() + 1):
            for c_ind in range(c.min(), c.max() + 1):
                expanded_high_sal_blobs[r_ind, c_ind] = 2

    processed_image_new = processed_image + expanded_high_sal_blobs

    final_high_salient_ind = np.argwhere(processed_image_new >= 3)
    final_middle_salient_ind = np.argwhere(processed_image_new == 1)
    final_non_salient_ind = np.argwhere((processed_image_new == 0) | (processed_image_new == 2))

    for i in final_high_salient_ind:
        processed_image_new[i[0], i[1]] = 2
    for i in final_middle_salient_ind:
        processed_image_new[i[0], i[1]] = 1
    for i in final_non_salient_ind:
        processed_image_new[i[0], i[1]] = 0

    # fig, ax = plt.subplots(1, 2, figsize=(16, 12))
    # ax[0].imshow(processed_image)
    # ax[1].imshow(processed_image_new)
    # plt.show()

    return processed_image_new


# return the perfect rectangles
def identfiy_perfect_rectangles_and_rectilinear_polygons(high_middle_zero_salient_image):
    temp_image_middle_sal = high_middle_zero_salient_image.copy()
    temp_image_high_sal = np.zeros((high_middle_zero_salient_image.shape[0], high_middle_zero_salient_image.shape[1]))

    high_sal_ind = np.argwhere(high_middle_zero_salient_image == 2)

    for i in high_sal_ind:
        temp_image_middle_sal[i[0], i[1]] = 0

    for i in high_sal_ind:
        temp_image_high_sal[i[0], i[1]] = 1

    # =====identify the perfect rectilinear regions with perfect rectangles and the imperfect rectangles for the <<highest>> salient regions
    all_labels = measure.label(temp_image_high_sal)
    blobs_labels = measure.label(all_labels, background=0, neighbors=4)

    unique, count = np.unique(blobs_labels, return_counts=True)

    unique = list(unique)
    count = list(count)

    perfect_rectangles_high_sal = []
    imperfect_rect_polygon_blobs_high_sal = []
    for i in range(1, len(unique)):
        perfect_rectangles_high_sal, imperfect_rect_polygon_blobs_high_sal = find_perfect_imperfect_rectangles(
            unique[i], blobs_labels,
            perfect_rectangles_high_sal,
            imperfect_rect_polygon_blobs_high_sal)
    # ====== identify the perfect rectilinear regions with perfect rectangles and the imperfect rectangles for the <<highest>> salient regions ends

    # ====== identify the perfect rectilinear regions with perfect rectangles and the imperfect rectangles for the <<middle>> salient regions
    all_labels = measure.label(temp_image_middle_sal)
    blobs_labels = measure.label(all_labels, background=0, neighbors=4)

    unique, count = np.unique(blobs_labels, return_counts=True)

    unique = list(unique)
    count = list(count)

    # identify whether a given remining middle salient area contins any single rectangles,polygons without holes, polygons with holes
    perfect_rectangles = []
    imperfect_rect_polygon_blobs = []
    for i in range(1, len(unique)):
        perfect_rectangles, imperfect_rect_polygon_blobs = find_perfect_imperfect_rectangles(unique[i], blobs_labels,
                                                                                             perfect_rectangles,
                                                                                             imperfect_rect_polygon_blobs)

    rect_polygon_with_holes = []
    rect_polygon_without_holes = []
    count_perfect_holes = 0
    count_imperfect_holes = 0
    for temp_blob in imperfect_rect_polygon_blobs:
        perfect_holes = []
        imperfect_holes = []
        grid = np.zeros(
            (high_middle_zero_salient_image.shape[0], high_middle_zero_salient_image.shape[1]), dtype=np.uint8)
        perfect_holes, imperfect_holes = detect_holes_in_rectilinear_polygon(temp_blob, grid, perfect_holes,
                                                                             imperfect_holes)

        count_perfect_holes += len(perfect_holes)
        count_imperfect_holes += len(imperfect_holes)

        # we consider only the perfect holes
        if len(perfect_holes) == 0:
            rect_polygon_without_holes.append(temp_blob)
        else:
            rect_polygon_with_holes.append([temp_blob, perfect_holes])

    # ====== identify the perfect rectilinear regions with perfect rectangles and the imperfect rectangles for the <<middle>> salient regions ends

    # fig, ax = plt.subplots(1, 2, figsize=(16, 12))
    # ax[0].imshow(high_middle_zero_salient_image)
    #
    # # plot the imperfect rectangles
    # grid = np.zeros(
    #     (high_middle_zero_salient_image.shape[0], high_middle_zero_salient_image.shape[1]))
    #
    # for bloc_id, block in enumerate(imperfect_rect_polygon_blobs):
    #     for coord in block:
    #         grid[coord[0], coord[1]] = 1
    #
    # # perfect_rectangle_grid = np.zeros(
    # #     (high_middle_zero_salient_image.shape[0], high_middle_zero_salient_image.shape[1]))
    # for block in perfect_rectangles:
    #     grid[block[0][0]:block[2][0], block[0][1]:block[2][1]] = 2
    #
    # for block in rect_polygon_with_holes:
    #     for holes in block[1]:
    #         grid[holes[0][0]:holes[3][0], holes[0][1]:holes[3][1]] = 3
    #
    # # for block in imperfect_holes:
    # #     grid[block[0][0]:block[3][0], block[0][1]:block[3][1]] = 4
    #
    # # ax[1].imshow(grid)
    # # plt.show()

    return perfect_rectangles, rect_polygon_without_holes, rect_polygon_with_holes, perfect_rectangles_high_sal, imperfect_rect_polygon_blobs_high_sal,count_perfect_holes,count_imperfect_holes


# find the whether the given polygon is perfect or imperfect rectangle
def find_perfect_imperfect_rectangles(polygon, blobs_labels, perfect_rectangles, imperfect_rectangle_blobs):
    temp_blob = np.argwhere(blobs_labels == polygon)
    temp_blob_list = list(temp_blob)

    # temp_blob_list.sort()
    r = temp_blob[:, 0]
    c = temp_blob[:, 1]

    r_min = r.min()
    r_max = r.max()
    c_min = c.min()
    c_max = c.max()

    blob_area = len(temp_blob_list)
    ideal_rectngle_area = ((r_max - r_min) + 1) * ((c_max - c_min) + 1)

    # if the ideal region is single column or single row
    if ideal_rectngle_area == 0:
        perfect_rectangles.append(
            [[r_min, c_min], [r_min, c_max + 1], [r_max + 1, c_max + 1], [r_max + 1, c_min]])
    # if the  ideal region is not a single column or single row
    else:
        # if the polygon is a perfect rectangle
        if blob_area == ideal_rectngle_area:
            perfect_rectangles.append(
                [[r_min, c_min], [r_min, c_max + 1], [r_max + 1, c_max + 1], [r_max + 1, c_min]])
        # if the polygon is not a perfect rectangle and may contain holes as well. not detected yet
        else:
            imperfect_rectangle_blobs.append(temp_blob)

    return perfect_rectangles, imperfect_rectangle_blobs


# detect the holes inside a rectilinear polygon
def detect_holes_in_rectilinear_polygon(polygon, grid, perfect_holes, imperfect_holes):
    for coord in polygon:
        grid[coord[0], coord[1]] = 1

    grid = np.abs(1 - grid)

    # detect blobs in the
    all_labels = measure.label(grid)
    blobs_labels = measure.label(all_labels, background=0, neighbors=4)

    unique, count = np.unique(blobs_labels, return_counts=True)

    unique = list(unique)
    count = list(count)

    # remove the background (which is the inverted case of the original rectilinear polygon)
    # remove the first partition polygon which typically has the highes area which is exactly the
    # original polygon background after the inversion.

    unique = unique[1:]
    count = count[1:]

    max_ind = count.index(max(count))
    count.remove(count[max_ind])
    unique.remove(unique[max_ind])

    if len(unique) > 0:
        for i in range(0, len(unique)):
            perfect_holes, imperfect_holes = find_perfect_imperfect_rectangles(unique[i],
                                                                               blobs_labels,
                                                                               perfect_holes,
                                                                               imperfect_holes)

    return perfect_holes, imperfect_holes


# function to identify the oov regions and extract them and split them
# fov_image cotains all the high (fol), middle (fov) and oov (black region)
def extract_oov(fov_image):
    oov_region = fov_image.copy()

    right = 0
    down = 1
    left = 2
    up = 3

    w = fov_image.shape[1]
    h = fov_image.shape[0]

    # fig, axs = plt.subplots(1, 1, figsize=(16, 12))
    # axs.imshow(fov_image)
    #
    # plt.show()

    # temporarly stopped coding this function. Way too complicated
    # bound_touch_cells = traverse_and_find_oov_tiles_next2_fov_boundary_touching_cells(fov_image=fov_image,
    #                                                                                   w=w,
    #                                                                                   h=h)

    # get the upper most left and lower most right boundary of the fov region
    upper_left = [0, 0]
    upper_left_found = False
    for r in range(h):
        for c in range(w):
            if fov_image[r, c] > 0:
                upper_left = [r, c]
                upper_left_found = True
                break
        if upper_left_found:
            break

    lower_right = [0, 0]
    lower_right_found = False
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if fov_image[r, c] > 0:
                lower_right = [r, c]
                lower_right_found = True
                break
        if lower_right_found:
            break

    # fill the left side
    left_limits = []
    right_limits = []
    for r in range(h):

        # each row identify the left and right indices which are limits for the left and right side filling
        left_found = False
        right_found = False
        for c in range(w):
            if not left_found and fov_image[r, c] > 0:
                left_limits.append(c)
                left_found = True
                upper_left[1] = c

            if left_found and (not right_found) and fov_image[r, c] == 0:
                right_limits.append(c)
                right_found = True
                lower_right[1] = c

        if (not left_found) and (not right_found):
            left_limits.append(upper_left[1])
            right_limits.append(upper_left[1])
        if left_found and not right_found:
            right_limits.append(w)

        if not left_found and right_found:
            left_limits.append(0)

    print(right_limits)
    for r in range(h):
        for c in range(left_limits[r]):
            if oov_region[r, c] == 0:
                oov_region[r, c] = 4

        # print(right_limits[r])
        for c in range(right_limits[r], w):

            if oov_region[r, c] == 0:
                oov_region[r, c] = 5

    oov_region = np.where(oov_region == 0, 6, oov_region)
    oov_region = np.where(oov_region == 1, 0, oov_region)
    oov_region = np.where(oov_region == 2, 0, oov_region)

    oov_region = np.flip(oov_region, axis=1).transpose()

    # fig, axs = plt.subplots(1, 2, figsize=(16, 12))
    # axs[0].imshow(fov_image)
    # axs[1].imshow(oov_region)
    # plt.show()

    return oov_region

# BELOW CODES ARE NOT IMPORTANT
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT *********
# # after identifying the rectilinear polygon, this function extract the indices of all those rectlinear
# # polygons. Then return them in 2 lists. For the outer regions, we don't consider the polygons with holes
# def take_rect_polygons(oov):
#     all_labels = measure.label(oov)
#     blobs_labels = measure.label(all_labels, background=0, neighbors=4)
#
#     unique, count = np.unique(blobs_labels, return_counts=True)
#
#     unique = list(unique)
#     count = list(count)
#
#     perfect_rectangles_oov = []
#     imperfect_rect_polygon_oov = []
#     for i in range(1, len(unique)):
#         perfect_rectangles_oov, imperfect_rect_polygon_oov = find_perfect_imperfect_rectangles(
#             unique[i], blobs_labels,
#             perfect_rectangles_oov,
#             imperfect_rect_polygon_oov)
#
#     return perfect_rectangles_oov,imperfect_rect_polygon_oov
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT *********
#
#
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
# # traverse through the boundary. Find the bundary touching fov cells and find the cell coordinates
# # of oov tiles next to those fov tiles.
# def traverse_and_find_oov_tiles_next2_fov_boundary_touching_cells(fov_image, w, h):
#     # Traverse around the image boundary a complete round and find if there are any boundary touching fov cells
#
#     # traverse right
#     bound_touch_cells_right = []
#     r = 0
#     for c in range(w):
#         if fov_image[r, c] > 0:
#             bound_touch_cells_right.append([r, c])
#
#     # remove consecutive boundary touching cells
#     cell_blocks = []
#     count = 0
#     single_block = []
#
#     bound_touch_cells_right = [[0, 1], [0, 2], [0, 4], [0, 6], [0, 7]]
#
#     # find the consecutive boundary touching cells and get only the terminal cells
#     for cell_ind in range(1, len(bound_touch_cells_right)):
#         # starting one single block
#         if count == 0:
#             single_block.append(bound_touch_cells_right[cell_ind - 1])
#         # check whether the current and previous cells are consectutiv. If the so, again check the current cell is the last cell,
#         # if so create a single block otherwise, continue for the next cell
#         if bound_touch_cells_right[cell_ind][1] - bound_touch_cells_right[cell_ind - 1][1] == 1:
#             if cell_ind == len(bound_touch_cells_right) - 1:
#                 single_block.append(bound_touch_cells_right[cell_ind])
#                 cell_blocks.append(single_block)
#             else:
#                 count += 1
#                 continue
#         else:
#             # If it is not a single block, add the current cell as the other edge and complete one single block
#             if count != 0:
#                 single_block.append(bound_touch_cells_right[cell_ind - 1])
#             cell_blocks.append(single_block)
#
#         # if not continued from above, a single block has been created.
#         # re-initialize the block and count
#         single_block = []
#         count = 0
#
#     # find the oov cells next to the above boundary touching coordinates
#     oov_next_tiles = []
#     for cell_block_ind in range(len(cell_blocks)):
#         block = cell_blocks[cell_block_ind]
#         # check for corner connectivity.
#         if len(block) == 1:
#             # check for corner corner
#             r, l, _, _ = check_for_corner_cell(cell=block[0],
#                                                is_hori_cell=True,
#                                                w=w,
#                                                h=h)
#             # if the single cell is right corner tile
#             # if r:
#             # oov_next_tiles.append([block[]])
#
#     print(1)
#     return
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
#
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
# # check whether the given cell is belongs to a corner or not
# def check_for_corner_cell(is_hori_cell, cell, w, h):
#     is_right = False
#     is_left = False
#     is_up = False
#     is_down = False
#
#     if is_hori_cell:
#         if cell[1] == w - 1:
#             is_right = True
#         if cell[1] == 0:
#             is_left = True
#     else:
#         if cell[0] == h - 1:
#             is_down = True
#         if cell[0] == 0:
#             is_up = True
#
#     return is_right, is_left, is_up, is_down
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
#
#
#
# # =========== not used functions ============#
# # =========== not used functions ============#
# def init_analysis_of_threshold_maps():
#     path = '/Users/ckat9988/Documents/Research/FoV/Feature_extraction/Spatil_saliency_detection/SalAlgo2_Panosal/new_vidoes_sal_maps'
#     videos = ['BasketBall_new', 'ChariotRace_new', 'DrivingWith_new', 'FootBall_new', 'GirlDance_new',
#               'HogRider_new', 'Kitchen_new', 'MagicShow_new', 'PerlisPanel_new', 'RollerCoster_new',
#               'SharkShipWreck_new', 'Skiing_new']
#     num_of_frames = 24
#     list = np.arange(0, 1795, 75)
#     for v in range(len(videos)):
#         print(v)
#         fig, ax = plt.subplots(num_of_frames, 4, figsize=(20, 75))
#         random_list = np.random.choice(list, num_of_frames, replace=False)
#         for ind, i in enumerate(list):
#             print(i)
#             im_path = path + '/' + videos[v] + '/' + 'frame' + str(i) + '.png'
#             im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
#
#             w = im.shape[1]
#             h = im.shape[0]
#             # test different re sampling experiments aplicable of the
#             down_sam_image_linear = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
#
#             ax[ind, 0].imshow(down_sam_image_linear)
#
#             ax[ind, 1].hist(down_sam_image_linear.ravel(), bins=256, range=(0.0, 255.0), fc='k',
#                             ec='k')  # calculating histogram
#             plt.title(str(i))
#
#             enhanced_img = cv2.equalizeHist(down_sam_image_linear)
#             ax[ind, 2].imshow(enhanced_img)
#
#             ax[ind, 3].hist(enhanced_img.ravel(), bins=256, range=(0.0, 255.0), fc='k', ec='k')  # calculating histogram
#             plt.title('histogram equalized')
#
#         plt.savefig(
#             '/Users/ckat9988/Documents/Research/FoV/Results/Sal_analysis/initial_sal_analysis_different_videos/' +
#             videos[v] + 'sal_map.pdf')
#         # plt.show()
#         plt.close()
#         plt.clf()
#         print(1)
#
#     # read images
#
#     return
#
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
# # algo to identify the threshold for the saliency map gnereation
# def define_threshold_for_sal_map():
#     im_path = '/Users/ckat9988/Documents/Research/FoV/Feature_extraction/Spatil_saliency_detection/SalAlgo2_Panosal/new_vidoes_sal_maps/DrivingWith_new/frame65.png'
#
#     im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
#
#     hori_tiles = 20
#     verti_tiles = 10
#
#     w = im.shape[1]
#     h = im.shape[0]
#     # test different re sampling experiments aplicable of the
#     down_sam_image_linear = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
#
#     new_w = down_sam_image_linear.shape[1]
#     new_h = down_sam_image_linear.shape[0]
#
#     tile_w = new_w // hori_tiles
#     tile_h = new_h // verti_tiles
#
#     # normalize the image
#     enhanced_img = cv2.equalizeHist(down_sam_image_linear)
#     enhanced_img = enhanced_img / np.max(enhanced_img)
#
#     P_t = 0.6
#     P_abs = 0.0
#     epsilon = 0.1
#
#     th_min = 0.6  # need to revise empirically
#     th_max = 1.0  # need to revise empirically
#
#     th_cur = 0.85
#     th_prev = 0
#
#     thresh_img = np.zeros((enhanced_img.shape[0], enhanced_img.shape[1]))
#     number_of_trials = 0
#
#     while (number_of_trials < 20):  # (((P_t - P_abs) > epsilon) or (P_t - P_abs < 0)
#
#         number_of_trials += 1
#
#         # threshold the image
#         coor = np.where(enhanced_img >= th_cur)
#
#         ind_1 = np.argwhere(enhanced_img >= th_cur)
#         ind_0 = np.argwhere(enhanced_img < th_cur)
#         # for i in ind_1:
#         thresh_img[ind_1[:, 0], ind_1[:, 1]] = 1
#         # for i in ind_0:
#         thresh_img[ind_0[:, 0], ind_0[:, 1]] = 0
#
#         plt.imshow(thresh_img)
#         plt.show()
#
#         # find how many basic tiles are transmitted to the client side
#         transmitted_tiles = 0
#
#         for i in range(verti_tiles - 1):
#             for j in range(hori_tiles - 1):
#                 sub_region = thresh_img[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
#                 # at the moment assume that if there is at least 1 pixel, we transmit that basic tile # need to revise
#                 if np.sum(sub_region) > 0:
#                     transmitted_tiles += 1
#
#         P_abs = (transmitted_tiles * tile_w * tile_h) / (new_w * new_h)
#         print(str(P_t - P_abs) + '   ' + str(th_cur))
#         th_prev = th_cur
#         # come to the exact optimum condition
#         if (P_t - P_abs) < epsilon:
#             print('optimum condition')
#             break
#
#         # If the transmitted pixels are very lower than the expected threshold decrease the
#         # pixel threshold and transmit more pixels
#         elif (P_t - P_abs) > epsilon:
#             print('abs more lower than threshold')
#             th_max = th_cur
#             th_cur = (th_min + th_cur) / 2
#
#         # If the transmitted pixels is greater than the expected threshold increase the
#         # pixel threshold and transmit less number of  pixels
#         elif (P_t - P_abs) < 0:
#             print('abs greater than threshold')
#             th_min = th_cur
#             th_cur = (th_max + th_cur) / 2
#
#     return
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
#
# # ****** NOT IMPORTANT *********
# # ****** NOT IMPORTANT ********
# # identify the independant regions of the saliency and keep only the largest area
# #  of the tile
# def identify_independent_clusters_keep_largest(im):
#     image = im.copy()
#
#     all_labels = measure.label(im)
#     blobs_labels = measure.label(all_labels, background=0, neighbors=4)
#
#     unique, count = np.unique(blobs_labels, return_counts=True)
#
#     unique = list(unique)
#     count = list(count)
#
#     unique = unique[1:]
#     count = count[1:]
#
#     max_count_ind = count.index(max(count))
#     max_count_unique_val = unique[max_count_ind]
#
#     ind = np.argwhere(blobs_labels != max_count_unique_val)
#
#     for i in ind:
#         im[i[0], i[1]] = 0
#
#     fig, ax = plt.subplots(1, 2, figsize=(16, 12))
#     ax[0].imshow(image)
#     ax[1].imshow(im)
#     plt.show()
#     return im
#
# # =========== not used functions ============#
# # =========== not used functions ============#
