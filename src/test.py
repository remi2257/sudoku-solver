import cv2
from src.grid_detector_img import get_lines_and_corners, preprocess_im
from src.fonctions import timer_decorator, my_resize
from src.settings import *


@timer_decorator
def function_test(im):
    print(im.shape)
    lines_raw = cv2.HoughLinesP(preprocess_im(im),
                                rho=hough_rho, theta=hough_theta,
                                threshold=thresh_hough_p,
                                minLineLength=minLineLength_h_p, maxLineGap=maxLineGap_h_p)


im = cv2.imread("images_test/sudoku5.jpg")
im2 = my_resize(im, width=600)
im3 = cv2.resize(im, (600,800))
function_test(im)
function_test(im2)
function_test(im3)
