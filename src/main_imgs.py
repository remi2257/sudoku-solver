import os
import time
import cv2
from tensorflow.keras.models import load_model
from tqdm import tqdm

from src.extract_n_solve.extract_digits import process_extract_digits
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grids
from src.extract_n_solve.new_img_generator import *
from src.useful_functions import my_resize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
save_path = "images_save/"


def process_1_img(im_path, model):
    resized = False
    frame = cv2.imread(im_path)
    if frame is None:
        return 2
    if frame.shape[0] > 1000:
        frame = my_resize(frame, width=900, height=900)
        resized = True
    im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(frame, resized=resized)
    if im_grids_final is None:
        return 3
    grids_matrix = process_extract_digits(im_grids_final, model,save_images_digit=False)
    if all(elem is None for elem in grids_matrix):
        return 4
    grids_solved = main_solve_grids(grids_matrix)

    if grids_solved is None:
        return 5

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids, list_transform_matrix)

    cv2.imwrite(os.path.dirname(im_path)+"/solved/"+
                os.path.splitext(os.path.basename(im_path))[0] + "_solved.jpg", im_final)

    return 1


if __name__ == '__main__':
    dataset_path = "dataset_test/"
    # dataset_path = "images_test/"
    my_model = load_model('model/my_model.h5')
    im_list_del = [dataset_path + im_path for im_path in os.listdir(dataset_path)
                   if im_path.endswith("solved.jpg") or im_path.startswith("grid_cut")]
    for im_path in im_list_del:
        os.remove(im_path)

    im_list = [dataset_path + im_path for im_path in os.listdir(dataset_path) if im_path.endswith(".jpg")]
    # im_list = sorted(im_list, key=lambda x: int(x[-7:-4]))
    results = 6 * [0]
    for im_path in tqdm(im_list, total=len(im_list)):
        # print(im_path)
        ret = process_1_img(im_path, my_model)
        results[0] += 1
        results[ret] += 1
        if ret == 3:
            print("Unable to Extract Grid", im_path)
        elif ret == 4:
            print("Unviable Grid:", im_path)

    time.sleep(0.5)
    print("Sucess Rate = {:.1f}% ({}/{})".format(100 * (results[1] / results[0]), results[1], results[0]))
    print(
        "Unable to Extract Grid Rate = {:.1f}% ({}/{})".format(100 * (results[3] / results[0]), results[3], results[0]))
    print("Unviable Grid Rate = {:.1f}% ({}/{})".format(100 * (results[4] / results[0]), results[4], results[0]))
