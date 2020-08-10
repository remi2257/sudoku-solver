import sys

from tensorflow.keras.models import load_model

from settings import *
from src.extract_n_solve.extract_digits import process_extract_digits
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grids
from src.extract_n_solve.new_img_generator import *
from src.useful_functions import my_resize

save_folder = "images_save/"
model_default_name = 'model/my_model.h5'


def process_single_img(frame, model, save=False):
    # Resizing image
    if frame.shape[0] > 1000 or frame.shape[0] < 800:
        old_shape = frame.shape
        frame = my_resize(frame, width=param_resize_width, height=param_resize_height)
    else:
        old_shape = None

    # Extracting grids
    im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(frame)
    if im_grids_final is None:
        return frame

    # Generate matrix representing digits in grids
    grids_matrix = process_extract_digits(im_grids_final, model)
    if all(elem is None for elem in grids_matrix):
        return frame

    # Solving grids
    grids_solved = main_solve_grids(grids_matrix)

    if grids_solved is None:
        return frame

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids, list_transform_matrix)

    if old_shape is not None:
        im_final = cv2.resize(im_final, old_shape[:2][::-1])

    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))[0] + "_solved.jpg", im_final)

    return im_final


if __name__ == '__main__':
    im_paths = [
        "images_test/sudoku.jpg",
    ]
    im_path = im_paths[0]
    res = process_single_img(frame=cv2.imread(im_path),
                             model=load_model(model_default_name),
                             save=False)

    cv2.imshow("res", res)
    cv2.waitKey(0)
