import logging
import os

import tensorflow
from cv2 import FONT_HERSHEY_SIMPLEX
from numpy import pi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# CAMERA

CAMERA_PORT = 1

# PATHS
my_dataset_path = "/media/hdd_linux/DataSet/Mine/"
temp_dataset_path = "/media/hdd_linux/DataSet/Mine/temp"
model_default_path = 'model/my_model.h5'

# ----TEXT DISPLAY----#
RED = (0, 0, 255)
PURPLE = (255, 0, 255)
ORANGE = (0, 127, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font_base = FONT_HERSHEY_SIMPLEX
font_scale_normal = 2.0
font_scale_small = 1.1
thickness_normal = 3
thickness_small = 2
# ----RESIZE----#
param_resize_height = 900
param_resize_width = 1600
resize_height_hough = 360
resize_width_hough = 640

output_width = 1365  # 853
output_height = 768  # 480

ratio_resize_hough = float(param_resize_height) / resize_height_hough

# ----PREPRO BIG IMAGE----#
block_size_big = 41
block_size_webcam_big = 21
mean_sub_big = 15
mean_sub_webcam_big = 5

# display_prepro_big = True
display_prepro_big = False

# ----GRID COUNTOURS----#
ratio_lim = 2
smallest_area_allow = 75000
approx_poly_coef = 0.1

# ----GRID UPDATE AND SIMILARITY----#
lim_apparition_not_solved = 12
lim_apparition_solved = 60
same_grid_dist_ratio = 0.05

target_h_grid, target_w_grid = 450, 450

# ----HOUGH----#
thresh_hough = 500
# thresh_hough_p = 170
# minLineLength_h_p = 5
# maxLineGap_h_p = 5
# hough_rho = 3
# hough_theta = 3 * pi / 180
thresh_hough_p = 100
minLineLength_h_p = 5
maxLineGap_h_p = 5
hough_rho = 3
hough_theta = 3 * pi / 180

# display_line_on_edges = True
display_line_on_edges = False

# ----PREPRO IMAGE DIGIT----#
block_size_grid = 29  # 43
block_size_webcam_grid = 25
mean_sub_grid = 25
mean_sub_webcam_grid = 5

# display_prepro_grid = True
display_prepro_grid = False

# ----DIGITS EXTRACTION----#
thresh_conf_cnn = 0.98
thresh_conf_cnn_high = 0.99
digits_2_check = 12

lim_bord = 10
thresh_h_low = 15
thresh_h_high = 50
thresh_area_low = 210
thresh_area_high = 900
l_case = 45
l_border = 1
offset_y = 2
min_digits_extracted = 13


# ----LOGGER----#


def logger_gen(level='DEBUG'):
    # In the Console
    # logging.basicConfig(format='%[levelname]s : %(message)s')
    logging.basicConfig(format='%(message)s')

    logger_obj = logging.getLogger(__name__)

    logger_obj.setLevel(level)

    # create a file handler
    handler = logging.FileHandler('logger.log')

    # create a logging format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(levelname)s - %(message)s')
    # handler.setFormatter(formatter)

    # add the handlers to the logger
    logger_obj.addHandler(handler)

    return logger_obj


logger = logger_gen()
