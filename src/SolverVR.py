import cv2

from settings import model_default_path, param_resize_width, param_resize_height
from src.useful_functions import my_resize

from src.extract_n_solve.grid_detector import GridDetector
from src.extract_n_solve.extract_digits import DigitsExtractor
from src.extract_n_solve.grid_solver import GridSolver
from src.extract_n_solve.new_img_generator import ImageGenerator


def decorator_resize(func):
    def wrapper(*args, **kwargs):
        img = kwargs["img"]
        if img.shape[0] > 1000 or img.shape[0] < 800:
            old_shape = img.shape
            kwargs["img"] = my_resize(img, width=param_resize_width, height=param_resize_height)
        else:
            old_shape = None

        im_final = func(*args, **kwargs)
        if old_shape is not None:
            im_final = cv2.resize(im_final, old_shape[:2][::-1])

        return im_final

    return wrapper


class SolverVR:
    def __init__(self, model_path=model_default_path):
        self.__grid_detector = GridDetector()
        self.__digits_extractor = DigitsExtractor(model_path=model_path)
        self.__grid_solver = GridSolver()
        self.__image_generator = ImageGenerator()

    @decorator_resize
    def solve_1_img(self, img, hint_mode=False):
        # Extracting grids
        unwraped_grid_list, points_grids, list_transform_matrix = self.__grid_detector.extract_grids(img)

        # Generate matrix representing digits in grids
        grids_matrix = self.__digits_extractor.process_imgs(unwraped_grid_list)

        # Solving grids
        grids_solved = self.__grid_solver.solve_grids(grids_matrix, hint_mode=hint_mode)

        # Creating image filled
        im_filled = self.__image_generator.create_image_filled(img,
                                                               unwraped_grid_list, grids_matrix, grids_solved,
                                                               points_grids, list_transform_matrix)

        return im_filled


if __name__ == '__main__':
    solver = SolverVR()

    main_ret = solver.solve_1_img(img=cv2.imread("images_test/sudoku.jpg"))

    cv2.imshow("Res", main_ret)
    cv2.waitKey(0)
