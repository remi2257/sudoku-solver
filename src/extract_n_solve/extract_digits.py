import cv2
import numpy as np

from settings import *
from src.solving_objects.Sudoku import verify_viable_grid


def find_adapted_thresh(preds_proba):
    confs = []
    for pred_proba in preds_proba:
        confs.append(max(pred_proba))
    confs = sorted(confs, reverse=True)
    best_confs = np.mean(confs[:digits_2_check])
    norm_conf = 10 * best_confs - 9
    if norm_conf >= thresh_conf_cnn_high:
        return thresh_conf_cnn_high

    return thresh_conf_cnn


def nothing(x):
    pass


def show_thresh(gray_enhance):
    while (1):
        # cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        A = max(3, 1 + 2 * cv2.getTrackbarPos('B1', 'track'))
        B = cv2.getTrackbarPos('M1', 'track')
        C = max(3, 1 + 2 * cv2.getTrackbarPos('B', 'track'))
        D = cv2.getTrackbarPos('M', 'track')
        adap = cv2.getTrackbarPos('M/G', 'track')
        blur_size = 2 * cv2.getTrackbarPos('blur_size', 'track') + 1
        if adap == 0:
            adap = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            adap = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        blurred = cv2.GaussianBlur(gray_enhance, (blur_size, blur_size), 0)
        thresh = cv2.adaptiveThreshold(gray_enhance, 255, adap, cv2.THRESH_BINARY, A, B)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, adap, cv2.THRESH_BINARY, C, D)

        cv2.imshow('thresh', thresh)
        cv2.imshow('thresh_with_blur', thresh2)


def show_trackbar_thresh(gray_enhance):
    max_block = 60
    max_mean = 80
    cv2.namedWindow('track')
    cv2.createTrackbar('B1', 'track', block_size_grid // 2 + 1, max_block, nothing)
    cv2.createTrackbar('M1', 'track', mean_sub_grid, max_mean, nothing)
    cv2.createTrackbar('B', 'track', block_size_grid // 2 + 1, max_block, nothing)
    cv2.createTrackbar('M', 'track', mean_sub_grid, max_mean, nothing)
    cv2.createTrackbar('M/G', 'track', 0, 1, nothing)
    cv2.createTrackbar('blur_size', 'track', 5, 10, nothing)
    show_thresh(gray_enhance)


def show_big_image(img, im_prepro, im_contours, pre_filled, display_annot=False):
    from src.useful_functions import my_resize
    color_text = (0, 0, 255)
    my_font = cv2.FONT_HERSHEY_SIMPLEX
    my_font_scale = 1.2
    m_thickness = 2

    top = np.concatenate((img, cv2.cvtColor(im_prepro, cv2.COLOR_GRAY2BGR)), axis=1)
    bot = np.concatenate((im_contours, pre_filled), axis=1)
    im_res = np.concatenate((top, bot), axis=0)

    if display_annot:
        h_im, w_im, _ = im_res.shape

        text1 = "0/ Initial Grid"
        text2 = "1/ Preprocessed Grid"
        text3 = "2/ Digits Detection"
        text4 = "3/ Digits Identification"

        (text_width, text_height) = cv2.getTextSize(text1, my_font, fontScale=my_font_scale, thickness=m_thickness)[0]
        cv2.rectangle(im_res, (0, 0),
                      (text_width + 15, text_height + 15),
                      BLACK, cv2.FILLED)
        cv2.putText(im_res, text1,
                    (5, text_height + 5),
                    my_font, my_font_scale, color_text, m_thickness)

        (text_width, text_height) = cv2.getTextSize(text2, my_font, fontScale=my_font_scale, thickness=m_thickness)[0]
        cv2.rectangle(im_res, (w_im // 2, 0),
                      (w_im // 2 + text_width + 15, text_height + 15),
                      BLACK, cv2.FILLED)
        cv2.putText(im_res, text2,
                    (w_im // 2 + 5, text_height + 5),
                    my_font, my_font_scale, color_text, m_thickness)

        (text_width, text_height) = cv2.getTextSize(text3, my_font, fontScale=my_font_scale, thickness=m_thickness)[0]
        cv2.rectangle(im_res, (0, h_im // 2),
                      (text_width + 15, h_im // 2 + text_height + 15),
                      BLACK, cv2.FILLED)
        cv2.putText(im_res, text3,
                    (5, h_im // 2 + text_height + 5),
                    my_font, my_font_scale, color_text, m_thickness)

        (text_width, text_height) = cv2.getTextSize(text4, my_font, fontScale=my_font_scale, thickness=m_thickness)[0]
        cv2.rectangle(im_res, (w_im // 2, h_im // 2),
                      (w_im // 2 + text_width + 15, h_im // 2 + text_height + 15),
                      BLACK, cv2.FILLED)
        cv2.putText(im_res, text4,
                    (w_im // 2 + 5, h_im // 2 + text_height + 5),
                    my_font, my_font_scale, color_text, m_thickness)

    cv2.imshow('res', my_resize(im_res, height=600))


def preprocessing_im_grid(img, is_gray=False):
    if is_gray:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    blurred = cv2.GaussianBlur(gray_enhance, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block_size_grid, mean_sub_grid)
    if display_prepro_grid:
        show_trackbar_thresh(gray_enhance)

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
            (text_width, text_height) = \
                cv2.getTextSize(digit, font, fontScale=font_scale_small, thickness=thickness_small)[0]
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale_small, (0, 3, 0), thickness_small * 3)
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale_small, (0, 0, 255), thickness_small)
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


def process_extract_digits(ims, model, display=False, display_digit=False,save_images_digit=False):
    grids = []
    # display_digit=True
    for img in ims:
        grids.append(process_extract_digits_single(img, model, display, display_digit=display_digit,
                                                   save_image_digits=save_images_digit))
        # cv2.waitKey(0)

    return grids


def process_extract_digits_single(img, model, display=False, display_digit=False, save_image_digits=False):
    # if resize:
    #     img = cv2.resize(img, (450, 450))
    # else:
    #     img = img
    h_im, w_im = img.shape[:2]
    im_prepro, gray_enhance = preprocessing_im_grid(img)
    im_contours = img.copy()
    contours, _ = cv2.findContours(im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_digits = []
    loc_digits = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_true, x_true = y + h / 2, x + w / 2
        if x_true < lim_bord or y_true < lim_bord or x_true > w_im - lim_bord or y_true > h_im - lim_bord:
            continue
        if thresh_h_low < h < thresh_h_high and thresh_area_low < w * h < thresh_area_high:
            if display:
                cv2.drawContours(im_contours, [cnt], -1, (0, 255, 0), 1)
                # print(w*h)
            y1, y2 = y - offset_y, y + h + offset_y
            border_x = max(1, int((y2 - y1 - w) / 2))
            x1, x2 = x - border_x, x + w + border_x
            # digit = im_prepro[y1:y2, x1:x2]
            digit_cut = gray_enhance[max(y1, 0):min(y2, h_im), max(x1, 0):min(x2, w_im)]
            _, digit_thresh = cv2.threshold(digit_cut,
                                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # digit_w_border = cv2.copyMakeBorder(digit, l_border, l_border, l_border, l_border,
            #                                     cv2.BORDER_CONSTANT, None, 255)
            img_digits.append(cv2.resize(digit_thresh, (28, 28), interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
            loc_digits.append([y_true, x_true])
    if not img_digits:
        if display:
            cv2.imshow("im_contours", im_contours)
        return None
    img_digits_np = np.array(img_digits) / 255.0
    preds_proba = model.predict(img_digits_np)

    # preds = np.argmax(preds_proba, axis=1) + 1
    preds = []
    nbr_digits_extracted = 0
    # adapted_thresh_conf_cnn = find_adapted_thresh(preds_proba)
    adapted_thresh_conf_cnn = thresh_conf_cnn
    for pred_proba in preds_proba:
        arg_max = np.argmax(pred_proba)
        if pred_proba[arg_max] > adapted_thresh_conf_cnn and arg_max<9:
            preds.append(arg_max + 1)
            nbr_digits_extracted += 1
        else:
            preds.append(-1)

    if display_digit:
        for i in range(len(preds)):
            y, x = loc_digits[i]
            cv2.imshow('pred_{} - {:.6f} - x/y : {}/{}'.format(preds[i], 100 * max(preds_proba[i]), int(x), int(y)),
                       img_digits[i])
    if save_image_digits:
        for i in range(len(preds)):
            pred = preds[i]
            img_digit = img_digits[i]
            class_name = "N" if pred == -1 else str(pred)
            target_folder = os.path.join(temp_dataset_path, class_name + "/")
            if not os.path.isdir(target_folder):
                os.mkdir(target_folder)
            name = "{}_{}.jpg".format(class_name,len(os.listdir(target_folder)))
            filename = os.path.join(target_folder, name)
            cv2.imwrite(filename, img_digit)

    if nbr_digits_extracted < min_digits_extracted:
        if display:
            cv2.imshow("im_contours", im_contours)
        return None
    grid = fill_numeric_grid(preds, loc_digits, h_im, w_im)

    if display:
        # print(grid)
        show_big_image(img, im_prepro, im_contours, fill_img_grid(img, grid))

    if verify_viable_grid(grid):
        return grid
    else:
        return None


if __name__ == '__main__':
    from tensorflow.keras.models import load_model

    model = load_model('model/my_model.h5')

    im_path = "images_test/grid_cut_0.jpg"
    # im_path = "images_save/023_failed.jpg"
    # im_path = "images_test/izi.png"
    im = cv2.imread(im_path)
    cv2.imshow("im", im)
    res_grids = process_extract_digits([im], model,
                                       display=True, display_digit=True)
    cv2.waitKey()
    # print(res_grids)
