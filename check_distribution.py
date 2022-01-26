
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches



def draw_tiles(vp_t,sel_ct,fetch_t, data_store_path, n):

    comb_viewport_map = np.zeros((540, 960))
    fig, ax = plt.subplots(1, 2, figsize=(40, 20))
    tile_w = 48
    tile_h = 54

    ax[0].imshow(comb_viewport_map)
    for rect_ind, rect in enumerate(sel_ct):
        h = np.abs(rect[2] * tile_h - rect[0] * tile_h)
        w = np.abs(rect[3] * tile_w - rect[1] * tile_w)

        rect = patches.Rectangle((rect[1] * tile_w, rect[0] * tile_h), width=w, height=h,
                                 facecolor='r',
                                 alpha=0.7,
                                 edgecolor='#ffffff', linewidth=3)
        ax[0].add_patch(rect)

    ax[0].imshow(comb_viewport_map)
    for rect_ind, rect in enumerate(fetch_t):
        h = np.abs(rect[2] * tile_h - rect[0] * tile_h)
        w = np.abs(rect[3] * tile_w - rect[1] * tile_w)

        rect = patches.Rectangle((rect[1] * tile_w, rect[0] * tile_h), width=w, height=h,
                                 facecolor='b',
                                 alpha=0.7,
                                 edgecolor='#ffffff', linewidth=3)
        ax[0].add_patch(rect)

    ax[1].imshow(comb_viewport_map)
    for rect_ind, rect in enumerate(vp_t):
        h = np.abs(rect[2] * tile_h - rect[0] * tile_h)
        w = np.abs(rect[3] * tile_w - rect[1] * tile_w)

        rect = patches.Rectangle((rect[1] * tile_w, rect[0] * tile_h), width=w, height=h,
                                 facecolor='g',
                                 alpha=0.6,
                                 edgecolor='k', linewidth=3)
        ax[1].add_patch(rect)

    plt.savefig(data_store_path + '/user_' + str(n) + '/sample.pdf')

    return





