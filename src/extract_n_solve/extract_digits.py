import cv2
import numpy as np

from settings import *
from src.solving_objects.Sudoku import Sudoku, verify_viable_grid
from tensorflow.keras.models import load_model

color = (0, 155, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2


class DigitsExtractor:
    def __init__(self, model_path):
        self.__tf_model = load_model(model_path)

    def process_imgs(self, list_im_grids):
        return [self.process_single_img(im_grid) for im_grid in list_im_grids]

    def process_single_img(self, img):
        h_im, w_im = img.shape[:2]
        im_prepro, gray_enhance = self.preprocessing_im_grid(img)
        contours, _ = cv2.findContours(im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_digits = []
        loc_digits = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            y_true, x_true = y + h / 2, x + w / 2
            if x_true < lim_bord or y_true < lim_bord or x_true > w_im - lim_bord or y_true > h_im - lim_bord:
                continue
            if thresh_h_low < h < thresh_h_high and thresh_area_low < w * h < thresh_area_high:
                y1, y2 = y - offset_y, y + h + offset_y
                border_x = max(1, int((y2 - y1 - w) / 2))
                x1, x2 = x - border_x, x + w + border_x
                digit_cut = gray_enhance[max(y1, 0):min(y2, h_im), max(x1, 0):min(x2, w_im)]
                _, digit_thresh = cv2.threshold(digit_cut,
                                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_digits.append(
                    cv2.resize(digit_thresh, (28, 28), interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
                loc_digits.append([y_true, x_true])
        if not img_digits:
            return None
        img_digits_np = np.array(img_digits) / 255.0
        preds_proba = self.__tf_model.predict(img_digits_np)

        preds = []
        nbr_digits_extracted = 0
        adapted_thresh_conf_cnn = thresh_conf_cnn
        for pred_proba in preds_proba:
            arg_max = np.argmax(pred_proba)
            if pred_proba[arg_max] > adapted_thresh_conf_cnn and arg_max < 9:
                preds.append(arg_max + 1)
                nbr_digits_extracted += 1
            else:
                preds.append(-1)

        if nbr_digits_extracted < min_digits_extracted:
            return None
        grid = self.fill_numeric_grid(preds, loc_digits, h_im, w_im)

        if verify_viable_grid(grid):
            return grid
        else:
            return None

    @staticmethod
    def preprocessing_im_grid(img, is_gray=False):
        if is_gray:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
        blurred = cv2.GaussianBlur(gray_enhance, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, block_size_grid, mean_sub_grid)

        return thresh, gray_enhance

    @staticmethod
    def fill_numeric_grid(preds, loc_digits, h_im, w_im):
        grid = np.zeros((9, 9), dtype=int)

        for pred, loc in zip(preds, loc_digits):
            if pred > 0:
                y, x = loc
                true_y = int(9 * y // h_im)
                true_x = int(9 * x // w_im)
                grid[true_y, true_x] = pred

        return grid


if __name__ == '__main__':
    im_path = "images_test/grid_cut_0.jpg"
    # im_path = "images_save/023_failed.jpg"
    # im_path = "images_test/izi.png"
    im = cv2.imread(im_path)
    cv2.imshow("im", im)
    extractor = DigitsExtractor(model_path='model/my_model.h5')
    res_grid = extractor.process_single_img(im)
    print(Sudoku(grid=res_grid))
    cv2.waitKey()
