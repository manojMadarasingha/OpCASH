# function which reades FoV data from different datasets.
# every funciton is corresponding to a specific dataset.
import glob

import pandas as pd
import settings


# read the orientation data from the distorted viweport maps from the individual settings.
def read_orientation_data_from_distorted_viewports(video_name):
    train_test = 'train'
    viewport_list_all_users = []

    # read first the closed set users
    # path_in = '/Users/ckat9988/Documents/Research/FoV/Inter_Process_Data/Viewport_maps/' \
    #           'indiviual_user_maps/' + train_test + '/' + 'closed' + '/' + video_name
    path_in = '/Users/ckat9988/Documents/Research/FoV/' \
              'Inter_Process_Data/Viewport_maps/ClosedSet_viewports/'+video_name


    user_count = settings.NUM_OF_USERS_TRAIN_CLOSED
    viewport_list_all_users.extend(read_data(user_count, path_in))

    # path_in = '/Users/ckat9988/Documents/Research/FoV/Inter_Process_Data/Viewport_maps/' \
    #           'indiviual_user_maps/' + train_test + '/' + 'open' + '/' + video_name
    # user_count = settings.NUM_OF_USERS_TEST_OPEN
    # viewport_list_all_users.extend(read_data(user_count, path_in))

    return viewport_list_all_users


def read_data(user_count, path_in):
    temp_list = []
    for u in range(user_count):  # user_count
        print('User ' + str(u))
        user_path = path_in + '/' + 'user_' + str(u)
        viewport_list = []
        for f in range(settings.NUM_OF_FRAMES):  # csv_count
            print('    Frame ' + str(f))
            viewport_list.append(pd.read_csv(user_path + '/' + 'frame_' + str(f) + '.csv', header=None).values)

        temp_list.append(viewport_list)

    return temp_list
