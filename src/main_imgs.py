from src.grid_solver import main_solve_grids
from src.grid_detector_img import main_grid_detector_img
from src.extract_digits import process_extract_digits
from src.new_img_generator import *
import os
import cv2
from keras.models import load_model
import tensorflow as tf
from src.fonctions import resize
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
save_path = "images_save/"


def process_1_img(im_path, model):
    resized = False
    frame = cv2.imread(im_path)
    if frame is None:
        return 2
    if frame.shape[0] > 1000:
        frame = resize(frame, height=900, width=900)
        resized = True
    im_grids_final, points_grids = main_grid_detector_img(frame, resized=resized)
    if im_grids_final is None:
        return 3
    grids_matrix = process_extract_digits(im_grids_final, model)
    if all(elem is None for elem in grids_matrix):
        return 4
    grids_solved = main_solve_grids(grids_matrix)

    if grids_solved is None:
        return 5

    ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
    im_final = recreate_img_filled(frame, ims_filled_grid, points_grids)

    cv2.imwrite(os.path.splitext(im_path)[0] + "_solved.jpeg", im_final)

    return 1


if __name__ == '__main__':
    dataset_path = "dataset_test/"
    model = load_model('model/my_model.h5')
    im_list = [dataset_path + im_path for im_path in os.listdir(dataset_path) if im_path.endswith(".jpg")]
    im_list = sorted(im_list, key=lambda x: int(x[-7:-4]))
    results = 6 * [0]
    for im_path in tqdm(im_list, total=len(im_list)):
        # print(im_path)
        ret = process_1_img(im_path, model)
        results[0] += 1
        results[ret] += 1

    print("Sucess Rate = {:.3f}% ({}/{})".format(100*(results[1] / results[0]), results[1], results[0]))
    print("Unviable Grid Rate = {:.3f}% ({}/{})".format(100*(results[4] / results[0]), results[4], results[0]))
