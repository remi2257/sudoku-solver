from src.grid_solver import main_solve_grids
from src.grid_detector_img import main_grid_detector_img
from src.extract_digits import process_extract_digits
from src.new_img_generator import *
import os
import cv2
from keras.models import load_model
import tensorflow as tf
import sys
import time
from src.fonctions import resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
save_folder = "images_save/"


def main_process_img(im_path, model, save=False, display=False):
    resized = False
    init = time.time()
    frame = cv2.imread(im_path)  # TODO Check if image not well oriented - EXIF data
    init0 = time.time()
    if frame is None:
        print("This path doesn't lead to a frame")
        sys.exit(3)
    if frame.shape[0] > 1000:
        frame = resize(frame, height=900, width=900)
        resized = True
    im_grids_final, points_grids = main_grid_detector_img(frame, display=display, resized=resized)
    found_grid_time = time.time()
    if im_grids_final is None:
        print("No grid found")
        sys.exit(3)
    grids_matrix = process_extract_digits(im_grids_final, model, display=display)
    if all(elem is None for elem in grids_matrix):
        print("Failed during extraction")
        sys.exit(3)
    extract_time = time.time()
    grids_solved = main_solve_grids(grids_matrix)

    if grids_solved is None:
        print(grids_matrix)
        cv2.imshow('grid_extract', im_grids_final[0])
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))[0] + "_failed.jpg", im_grids_final[0])
        cv2.waitKey()
        sys.exit(3)

    solve_time = time.time()

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids)
    final_time = time.time()

    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))[0] + "_solved.jpg", im_final)

    total_time = final_time - init

    load_time = init0 - init
    print("Load everything \t{:.1f}% - {:.3f}s".format(100 * load_time / total_time, load_time))
    founding_time = found_grid_time - init0
    print("Grid Research \t\t{:.1f}% - {:.3f}s".format(100 * founding_time / total_time, founding_time))
    extraction_duration = extract_time - found_grid_time
    print(
        "Digits Extraction \t{:.1f}% - {:.3f}s".format(100 * extraction_duration / total_time, extraction_duration))
    solving_duration = solve_time - extract_time
    print("Grid Solving \t\t{:.1f}% - {:.3f}s".format(100 * solving_duration / total_time, solving_duration))
    recreation_duration = final_time - solve_time
    print(
        "Image recreation \t{:.1f}% - {:.3f}s".format(100 * recreation_duration / total_time, recreation_duration))
    print("EVERYTHING DONE \t{:.2f}s".format(total_time))
    # print(grid)
    # print(grid_solved)

    if len(ims_filled_grid) == 1:
        cv2.imshow('im', frame)
        cv2.imshow('grid_extract', im_grids_final[0])
        cv2.imshow('grid_filled', ims_filled_grid[0])
    cv2.imshow('im_final', im_final)
    cv2.waitKey()


if __name__ == '__main__':
    model = load_model('model/my_model.h5')
    im_paths = [
        "images_test/sudoku.jpg",
        "images_test/sudoku1.jpg",
        "images_test/sudoku2.jpg",
        "images_test/izi_distord.jpg",
        "images_test/imagedouble.jpg",  # 4
        "images_test/sudoku5.jpg",
        "images_test/sudoku6.jpg",
        "dataset_test/004.jpg",  # 7
        "images_test/video_stop.png",  # 8
        "tmp/027.jpg",  # 9
    ]
    im_path = im_paths[7]
    main_process_img(im_path, model, save=False, display=True)
