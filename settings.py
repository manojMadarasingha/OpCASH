DATASET_1 = 1
DATASET_2 = 2
DATASET_3 = 3
DATASET_4 = 4
DATASET_5 = 5

HORI_TILES = 20
VERTI_TILES = 10

NUM_OF_FRAMES = 120
NUM_OF_USERS_TRAIN_CLOSED = 20
NUM_OF_USERS_TRAIN_OPEN = 10
NUM_OF_USERS_TEST_CLOSED = 20
NUM_OF_USERS_TEST_OPEN = 10
NUM_OF_TOT_USERS = 30

NUM_OF_USERS_APPROXI_THRESH_VALID = 17

# Tile coordinate indices after the initial MNC algorithm
L_L_M = 2
L_L_N = 3
U_R_M = 6
U_R_N = 7
AVG_PIXEL_INT_PER_TILE = 10
FRAME_NUM = 1
SAL_STATE = 12

# tile coordinates for the combined FOV, Buffer and OOV files
L_L_M_COMB = 0
L_L_N_COMB = 1
U_R_M_COMB = 2
U_R_N_COMB = 3
AVG_SAL_COMB = 4
SAL_STATE_COMB = 5

APPROXI_THRESH = 0.5
FINER_THRESH = 0.9

LEFT_ALIGNED = 0
CENTER_ALIGNED = 1
RIGHT_ALIGNED = 2

UPPER_ALIGNED = 0
CENTER_ALIGNED = 1
LOWER_ALIGNED = 2

FRAME_ERROR = -1
FRAME_OK = 1

TILE_W = 48
TILE_H = 54

FRAME_W = 960
FRAME_H = 540

RES_HD = [1920, 1080]
RES_4K = [3840, 2160]

# all the videos used
video_names = [
    "6_2",
    "6_4",
    "6_5",
    "6_8",
    "6_10",
    "6_11",
    "6_12",
    # "6_13",
    "6_17",
    "6_21",
    "6_25",
    "6_29",
    "ChariotRace_new",
    "Diving",
    "DrivingWith_new",
    "FootBall_new",
    "HogRider_new",
    "Kangarooisland",
    "Kitchen_new",
    "MagicShow_new",
    "PerlisPanel",
    "Rhinos2",
    "Rollercoaster1",
    "RollerCoster_new",
    "SFRsport",
    "SharkShipWreck_new",
    "Skiing_new",
    "Tahitisurf",
    "Timelapse",
    "WeirdAl"
]

# videos used for the train purposes and parameter decisioning
vid_train = ['6_11',
             '6_29',
             'DrivingWith_new',
             'HogRider_new',
             'Rollercoaster1',
             "6_2",
             '6_8',
             '6_25',
             'Kangarooisland',
             "Skiing_new",
             "6_4",
             "6_17",
             "SharkShipWreck_new",
             'Tahitisurf',
             "6_21",
             "SFRsport",
             "6_5",
             "6_12",
             "PerlisPanel",
             "Kitchen_new",
             ]

# videos used for the test purposes.
# data related to these videos are avilable in Github repo
vid_test = ['ChariotRace_new',
            'RollerCoster_new',
            'FootBall_new',
            "MagicShow_new",
            "WeirdAl",
            "6_10",
            "Rhinos2",
            "Timelapse",
            "Diving",

            ]

video_static_focus = [
    "6_4",
    "6_17",
    "SharkShipWreck_new",
    'Tahitisurf',
    "MagicShow_new"
]

video_moving_focus = [
    "6_10",
    "6_21",
    "Rhinos2",
    "SFRsport",
]

video_misc = [
    "6_5",
    "6_12",
    "Diving",
    "PerlisPanel",
    "Timelapse",
    "Kitchen_new",
    "WeirdAl",
]

user_order_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29]
user_order_2 = [27, 3, 18, 22, 11, 20, 23, 9, 24, 2, 10, 26, 5, 29, 25, 6, 8, 15, 12, 13, 17, 14, 21, 16, 28, 19, 1, 7,
                0, 4]
user_order_3 = [18, 10, 7, 6, 20, 17, 8, 0, 1, 22, 14, 28, 23, 2, 15, 25, 29, 16, 26, 19, 21, 5, 12, 9, 24, 27, 11, 3,
                13, 4]
user_order_4 = [19, 27, 3, 24, 4, 28, 8, 11, 0, 29, 17, 13, 23, 21, 26, 16, 1, 25, 2, 22, 18, 12, 20, 5, 7, 10, 6, 15,
                9, 14]
user_order_5 = [28, 26, 19, 24, 0, 8, 29, 2, 4, 11, 20, 3, 17, 23, 5, 27, 13, 10, 6, 16, 22, 14, 15, 25, 12, 21, 1, 18,
                7, 9]
rand_set = [user_order_1, user_order_2, user_order_3, user_order_4, user_order_5]
video_riding = [
    '6_11',
    '6_29',
    'ChariotRace_new',
    'DrivingWith_new',
    'HogRider_new',
    'Rollercoaster1',
    'RollerCoster_new']

video_explore = [
    "6_2",
    '6_8',
    '6_25',
    'FootBall_new',
    'Kangarooisland',
    "Skiing_new",
]
