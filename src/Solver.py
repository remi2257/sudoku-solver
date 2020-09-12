from settings import model_default_path, param_resize_width, param_resize_height
from src.extract_n_solve.extract_digits import process_extract_digits
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grids
from src.extract_n_solve.new_img_generator import write_solved_grids, recreate_img_filled
from src.useful_functions import my_resize

import cv2
from tensorflow.keras.models import load_model


class Solver:
    def __init__(self, model_path=model_default_path):
        self.__tf_model = load_model(model_path)

    def solve_1_img(self, img, hint_mode=False):
        # Resizing image
        if img.shape[0] > 1000 or img.shape[0] < 800:
            old_shape = img.shape
            img = my_resize(img, width=param_resize_width, height=param_resize_height)
        else:
            old_shape = None

        # Extracting grids
        im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(img)
        if im_grids_final is None:
            return img
        # Generate matrix representing digits in grids
        grids_matrix = process_extract_digits(im_grids_final, self.__tf_model)
        if all(elem is None for elem in grids_matrix):
            return img

        # Solving grids
        grids_solved = main_solve_grids(grids_matrix, hint_mode=hint_mode)

        if grids_solved is None:
            return img
        ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
        im_final = recreate_img_filled(img, ims_filled_grid, points_grids, list_transform_matrix)

        if old_shape is not None:
            im_final = cv2.resize(im_final, old_shape[:2][::-1])

        return im_final


if __name__ == '__main__':
    solver = Solver()

    solver.solve_1_img(img=cv2.imread("images_test/sudoku.jpg"))
