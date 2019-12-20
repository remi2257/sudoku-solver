import sys
import time

from tensorflow.keras.models import load_model

from settings import *
from src.extract_n_solve.extract_digits import process_extract_digits
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grids
from src.extract_n_solve.new_img_generator import *
from src.useful_functions import my_resize

save_folder = "images_save/"


def main_process_img(im_path, model, save=False, display=False, use_hough=True, save_images_digit=False):
    init = time.time()
    frame = cv2.imread(im_path)  # TODO Check if image not well oriented - EXIF data
    init0 = time.time()
    if frame is None:
        logger.error("This path doesn't lead to a frame")
        sys.exit(3)
    if frame.shape[0] > 1000 or frame.shape[0] < 800:
        frame = my_resize(frame, width=param_resize_width, height=param_resize_height)
    im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(frame, display=display,
                                                                                 use_hough=use_hough)
    found_grid_time = time.time()
    if im_grids_final is None:
        logger.error("No grid found")
        sys.exit(3)
    logger.info("Grid(s) found")
    grids_matrix = process_extract_digits(im_grids_final, model, display=display, display_digit=False,
                                          save_images_digit=save_images_digit)
    if all(elem is None for elem in grids_matrix):
        logger.error("Failed during digits extraction")
        sys.exit(3)
    logger.info("Extraction done")
    extract_time = time.time()
    grids_solved = main_solve_grids(grids_matrix)
    logger.info("Solving done")

    if grids_solved is None:
        print(grids_matrix)
        cv2.imshow('grid_extract', im_grids_final[0])
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))[0] + "_failed.jpg", im_grids_final[0])
        cv2.waitKey()
        sys.exit(3)

    solve_time = time.time()

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids, list_transform_matrix)
    final_time = time.time()

    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))[0] + "_solved.jpg", im_final)

    total_time = final_time - init

    load_time = init0 - init
    logger.info("Load Image\t\t\t{:03.1f}% - {:05.2f}ms".format(100 * load_time / total_time, 1000 * load_time))
    founding_time = found_grid_time - init0
    logger.info(
        "Grid Research \t\t{:03.1f}% - {:05.2f}ms".format(100 * founding_time / total_time, 1000 * founding_time))
    extraction_duration = extract_time - found_grid_time
    logger.info(
        "Digits Extraction \t{:03.1f}% - {:05.2f}ms".format(100 * extraction_duration / total_time,
                                                            1000 * extraction_duration))
    solving_duration = solve_time - extract_time
    logger.info(
        "Grid Solving \t\t{:03.1f}% - {:05.2f}ms".format(100 * solving_duration / total_time, 1000 * solving_duration))
    recreation_duration = final_time - solve_time
    logger.info(
        "Image recreation \t{:03.1f}% - {:05.2f}ms".format(100 * recreation_duration / total_time,
                                                           1000 * recreation_duration))
    logger.info("PROCESS DURATION \t{:.2f}s".format(final_time - init0))
    logger.info("EVERYTHING DONE \t{:.2f}s".format(total_time))
    # print(grid)
    # print(grid_solved)

    if len(ims_filled_grid) == 1:
        cv2.imshow('img', frame)
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
        "dataset_test/001.jpg",  # 7
        "images_test/video_stop.png",  # 8
        "tmp/035.jpg",  # 9
    ]
    im_path = im_paths[0]
    main_process_img(im_path, model, save=False, display=True)
