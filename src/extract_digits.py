import cv2
import numpy as np
import os
from keras.models import load_model
import tensorflow as tf
from src.Sudoku import verify_viable_grid

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

thresh_conf_cnn = 0.999


def show_thresh(gray_enhance):
    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        A = max(3, 1 + 2 * cv2.getTrackbarPos('B1', 'track'))
        B = cv2.getTrackbarPos('M1', 'track')
        C = max(3, 1 + 2 * cv2.getTrackbarPos('B', 'track'))
        D = cv2.getTrackbarPos('M', 'track')
        adap = cv2.getTrackbarPos('M/G', 'track')
        if adap == 0:
            adap = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            adap = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        blurred = cv2.GaussianBlur(gray_enhance, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(gray_enhance, 255, adap, cv2.THRESH_BINARY, A, B)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, adap, cv2.THRESH_BINARY, C, D)

        cv2.imshow('thresh', thresh)
        cv2.imshow('thresh2', thresh2)


def show_trackbar(gray_enhance):
    max_block = 40
    max_mean = 20
    cv2.namedWindow('track')
    cv2.createTrackbar('B1', 'track', 10, max_block, show_thresh)
    cv2.createTrackbar('M1', 'track', 13, max_mean, show_thresh)
    cv2.createTrackbar('B', 'track', 10, max_block, show_thresh)
    cv2.createTrackbar('M', 'track', 13, max_mean, show_thresh)
    cv2.createTrackbar('M/G', 'track', 0, 1, show_thresh)
    show_thresh(gray_enhance)


def preprocess_im(im, is_gray=False):
    if is_gray:
        gray = im
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    blurred = cv2.GaussianBlur(gray_enhance, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43, 12)
    # show_trackbar_adap_thresh(gray_enhance)

    return thresh, gray_enhance

    # _, thresh = cv2.threshold(gray_enhance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray_enhance, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
    # thresh = cv2.adaptiveThreshold(gray_enhance, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    # cv2.imshow('gray', gray)
    # cv2.imshow('gray_enhance', gray_enhance)
    #
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('thresh_2', thresh_2)
    # cv2.imshow('closing', closing)
    # cv2.imshow('opening', opening)
    # cv2.waitKey()
    # return opening


color = (0, 155, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2


def fill_img_grid(img, grid_matrix):
    im_filled_grid = img.copy()
    h_im, w_im = img.shape[:2]
    for y in range(9):
        for x in range(9):
            digit = str(grid_matrix[y, x])
            if digit == '0':
                continue
            true_y, true_x = int((y + 0.2) * h_im / 9), int((x + 0.2) * w_im / 9)
            (text_width, text_height) = cv2.getTextSize(digit, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale, (0, 3, 0), thickness * 3)
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale, (0, 0, 255), thickness)
    return im_filled_grid


def fill_numeric_grid(preds, loc_digits, h_im, w_im):
    grid = np.zeros((9, 9), dtype=int)

    for pred, loc in zip(preds, loc_digits):
        if pred > 0:
            y, x = loc
            true_y = int(9 * y // h_im)
            true_x = int(9 * x // w_im)
            grid[true_y, true_x] = pred

    return grid


thresh_offset = 5
thresh_h_low = 10
thresh_h_high = 50
thresh_area_low = 210
thresh_area_high = 900
l_case = 45
l_border = 1
offset_y = 2
min_digits_extracted = 20


def process_extract_digits(ims, model, display=False, display_digit=False):
    grids = []
    for im in ims:
        grids.append(process_extract_digits_single(im,
                                                   model, display, display_digit))

    return grids


def process_extract_digits_single(im, model, display=False, display_digit=False):
    # if resize:
    #     im = cv2.resize(img, (450, 450))
    # else:
    #     im = img
    h_im, w_im = im.shape[:2]
    im_prepro, gray_enhance = preprocess_im(im)
    im_contours = im.copy()
    contours, _ = cv2.findContours(im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_digits = []
    loc_digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # if abs((x + w / 2) // l_case) < thresh_offset or abs((y + h / 2) // l_case) < thresh_offset:
        #     cv2.drawContours(im_contours, [cnt], -1, (255, 255, 0), 1)
        #     continue
        if thresh_h_low < h < thresh_h_high and thresh_area_low < w * h < thresh_area_high:
            if display:
                cv2.drawContours(im_contours, [cnt], -1, (0, 255, 0), 1)
                # print(w*h)
            y1, y2 = y - offset_y, y + h + offset_y
            border_x = max(1, int((y2 - y1 - w) / 2))
            x1, x2 = x - border_x, x + w + border_x
            # digit = im_prepro[y1:y2, x1:x2]
            _, digit = cv2.threshold(gray_enhance[max(y1, 0):min(y2, h_im), max(x1, 0):min(x2, w_im)],
                                     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # digit_w_border = cv2.copyMakeBorder(digit, l_border, l_border, l_border, l_border,
            #                                     cv2.BORDER_CONSTANT, None, 255)
            img_digits.append(cv2.resize(digit, (28, 28)).reshape(28, 28, 1))
            loc_digits.append([(y1 + y2) / 2, (x1 + x2) / 2])
    img_digits_np = np.array(img_digits) / 255.0
    if not img_digits:
        return None
    preds_proba = model.predict(img_digits_np)

    # preds = np.argmax(preds_proba, axis=1) + 1
    preds = []
    nbr_digits_extracted = 0
    for pred_proba in preds_proba:
        arg_max = np.argmax(pred_proba)
        if pred_proba[arg_max] > thresh_conf_cnn:
            preds.append(arg_max + 1)
            nbr_digits_extracted += 1
        else:
            preds.append(-1)

    if nbr_digits_extracted < min_digits_extracted:
        return None
    grid = fill_numeric_grid(preds, loc_digits, h_im, w_im)
    if display_digit:
        for i in range(len(preds)):
            cv2.imshow('pred_' + str(preds[i]) + "-" + str(max(preds_proba[i])), img_digits[i])
        print(grid)
        cv2.imshow('im', im)
        cv2.imshow('im_prepro', im_prepro)
        cv2.imshow('contours', im_contours)
        cv2.imshow('pre-filled', fill_img_grid(im, grid))

        cv2.waitKey()
    if verify_viable_grid(grid):
        return grid
    else:
        return None


if __name__ == '__main__':
    model = load_model('model/my_model.h5')

    im_path = "images_test/grid_cut_0.jpg"
    # im_path = "images_save/023_failed.jpg"
    # im_path = "images_test/izi.png"
    img = cv2.imread(im_path)
    res_grids = process_extract_digits([img], model, display=True)
    print(res_grids)
