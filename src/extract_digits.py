import cv2
import numpy as np
import os
from keras.models import load_model
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

offset_y = 2
thresh_conf_cnn = 0.97


def preprocess_im(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # gray_enhance = 255 * (gray - gray.min()) / (gray.max() - gray.min())
    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))

    # _, thresh = cv2.threshold(gray_enhance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray_enhance, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('gray', gray)
    # cv2.imshow('gray_enhance', gray_enhance)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    # cv2.waitKey()
    return opening


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
thresh_area_low = 250
thresh_area_high = 800
l_case = 45
l_border = 5


def process_extract_digits(ims, model, display=False):
    # if resize:
    #     im = cv2.resize(img, (450, 450))
    # else:
    #     im = img
    grids = []
    for im in ims:
        h_im, w_im = im.shape[:2]

        thresh = preprocess_im(im)
        im_contours = im.copy()
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                offset_x = max(1, int((h - w) / 2))
                x1, x2 = x - offset_x, x + w + offset_x
                y1, y2 = y - offset_y, y + h + offset_y
                digit = thresh[y1:y2, x1:x2]
                digit_w_border = cv2.copyMakeBorder(digit, l_border, l_border, l_border, l_border,
                                                    cv2.BORDER_CONSTANT, None, 255)
                img_digits.append(cv2.resize(digit_w_border, (28, 28)).reshape(28, 28, 1))
                loc_digits.append([(y1 + y2) / 2, (x1 + x2) / 2])
        img_digits_np = np.array(img_digits) / 255.0
        if not img_digits:
            continue
        preds_proba = model.predict(img_digits_np)

        # preds = np.argmax(preds_proba, axis=1) + 1
        preds = []
        for pred_proba in preds_proba:
            arg_max = np.argmax(pred_proba)
            if pred_proba[arg_max] > thresh_conf_cnn:
                preds.append(arg_max + 1)
            else:
                preds.append(-1)

        grid = fill_numeric_grid(preds, loc_digits, h_im, w_im)
        if display:
            for i in range(len(preds)):
                cv2.imshow('pred_' + str(preds[i]) + "-" + str(max(preds_proba[i])), img_digits[i])
                # cv2.waitKey()
            print(grid)
            cv2.imshow('im', im)
            cv2.imshow('thresh', thresh)
            cv2.imshow('contours', im_contours)

            cv2.waitKey()

        grids.append(grid)
    return grids


if __name__ == '__main__':
    model = load_model('model/model_99_3.h5')

    im_path = "images/grid_cut_1.jpg"
    # im_path = "images/izi.png"
    img = cv2.imread(im_path)
    res_grids = process_extract_digits(img, model, display=True)
    print(res_grids)
