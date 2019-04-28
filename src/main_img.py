from src.grid_solver import main_solve_grid
from src.grid_detector import main_grid_detector_img
from src.extract_digits import process_extract_digits
from src.new_img_generator import *
import os
import cv2
from keras.models import load_model
import tensorflow as tf
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def main_process_img(im_path,save=False):
    model = load_model('model/model_99_3.h5')
    frame = cv2.imread(im_path)
    im_grid_final, M = main_grid_detector_img(frame)
    print("Grid Found")
    grid = process_extract_digits(im_grid_final, model)
    print("Digits Extracted")
    grid_solved = main_solve_grid(grid)

    if grid_solved is None:
        sys.exit(3)

    im_filled_grid = im_grid_final.copy()
    write_solved_grid(im_filled_grid, grid, grid_solved)
    im_final = recreate_img(frame, im_filled_grid, M)

    if save:
        cv2.imwrite(os.path.splitext(im_path)[0] + "_solved.jpg", im_final)

    else:
        print(grid_solved)
        cv2.imshow('im', frame)
        cv2.imshow('grid_extract', im_grid_final)
        cv2.imshow('grid_filled', im_filled_grid)
        cv2.imshow('im_final', im_final)
        cv2.waitKey()


if __name__ == '__main__':
    im_path = "images/sudoku.jpg"
    main_process_img(im_path)
