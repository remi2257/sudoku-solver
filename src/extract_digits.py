import cv2
import numpy as np
import os
from keras.models import load_model
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

thresh_contours_low = 800
thresh_contours_high = 3500
offset_y = 2
thresh_conf_cnn = 0.93

def preprocess_im(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def fill_numeric_grid(preds, loc_digits, h_im, w_im):
    grid = np.zeros((9, 9), dtype=int)

    for pred, loc in zip(preds, loc_digits):
        if pred > 0:
            y, x = loc
            true_y = int(9 * y // h_im)
            true_x = int(9 * x // w_im)
            grid[true_y, true_x] = pred

    return grid


def process_extract_digits(img, model, display=False, resize=False):
    if resize:
        im = cv2.resize(img, (900, 900))
    else:
        im = img
    h_im, w_im = im.shape[:2]
    l_border = 5

    thresh = preprocess_im(im)
    im_contours = im.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_digits = []
    loc_digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if thresh_contours_low < w * h < thresh_contours_high:
            if display:
                cv2.drawContours(im_contours, [cnt], -1, (0, 255, 0), 3)
            offset_x = max(1, int((h - w) / 2))
            x1, x2 = x - offset_x, x + w + offset_x
            y1, y2 = y - offset_y, y + h + offset_y
            digit = thresh[y1:y2, x1:x2]
            digit_w_border = cv2.copyMakeBorder(digit, l_border, l_border, l_border, l_border,
                                                cv2.BORDER_CONSTANT, None, 255)
            img_digits.append(cv2.resize(digit_w_border, (28, 28)).reshape(28, 28, 1))
            loc_digits.append([(y1 + y2) / 2, (x1 + x2) / 2])

    img_digits_np = np.array(img_digits) / 255.0
    preds_proba = model.predict(img_digits_np)

    # preds = np.argmax(preds_proba, axis=1) + 1
    preds = []
    for pred_proba in preds_proba:
        arg_max = np.argmax(pred_proba)
        if pred_proba[arg_max] >thresh_conf_cnn:
            preds.append(arg_max+1)
        else :
            preds.append(-1)

    grid = fill_numeric_grid(preds, loc_digits, h_im, w_im)
    if display:
        for i in range(len(preds)):
            cv2.imshow('pred_' + str(preds[i]) + "-" + str(max(preds_proba[i])), img_digits[i])
            # cv2.waitKey()
        print(grid)
        cv2.imshow('thresh', thresh)
        cv2.imshow('contours', im_contours)
        cv2.imshow('im', im)

        cv2.waitKey()
    return grid


if __name__ == '__main__':
    model = load_model('model/model_99_3.h5')

    im_path = "images/sudoku1_failed.jpg"
    # im_path = "images/izi.png"
    img = cv2.imread(im_path)
    grid=process_extract_digits(img, model, display=True)
    print(grid)