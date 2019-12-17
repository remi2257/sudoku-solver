import cv2

from settings import *
from src.solving_objects.MyHoughLines import *
from src.solving_objects.MyHoughPLines import *
from src.useful_functions import my_resize


# from src.fonctions import timer_decorator
# import time


def nothing(x):
    pass


def show_big_image(frame_resize, prepro_im, img_lines, img_contour, use_hough=False):
    if not use_hough:
        im_res = np.concatenate((frame_resize, img_contour), axis=0)
    else:
        top = np.concatenate((frame_resize, cv2.cvtColor(prepro_im, cv2.COLOR_GRAY2BGR)), axis=1)
        bot = np.concatenate((cv2.cvtColor(img_lines, cv2.COLOR_GRAY2BGR), img_contour), axis=1)
        im_res = np.concatenate((top, bot), axis=0)
        h_im, w_im, _ = im_res.shape

        text1 = "0/ Initial Image"
        text2 = "1/ Preprocessed Image"
        text3 = "2/ Hough Transform"
        text4 = "3/ Grids Extraction"

        (text_width, text_height) = cv2.getTextSize(text1, font, fontScale=font_scale_normal, thickness=thickness_normal)[0]
        # cv2.putText(im_res, text1,
        #             (w_im // 2 - text_width - 30, h_im // 2 - 30),
        #             font, font_scale_normal, WHITE, thickness_normal * 3)
        cv2.rectangle(im_res, (0, 0),
                      (text_width + 30, text_height + 30),
                      WHITE, cv2.FILLED)
        cv2.putText(im_res, text1,
                    (10, text_height + 10),
                    font, font_scale_normal, RED, thickness_normal)

        (text_width, text_height) = cv2.getTextSize(text2, font, fontScale=font_scale_normal, thickness=thickness_normal)[0]
        cv2.rectangle(im_res, (w_im // 2, 0),
                      (w_im // 2 + text_width + 30, text_height + 30),
                      WHITE, cv2.FILLED)
        cv2.putText(im_res, text2,
                    (w_im // 2 + 10, text_height + 10),
                    font, font_scale_normal, RED, thickness_normal)

        (text_width, text_height) = cv2.getTextSize(text3, font, fontScale=font_scale_normal, thickness=thickness_normal)[0]
        cv2.rectangle(im_res, (0, h_im // 2),
                      (text_width + 30, h_im // 2 + text_height + 30),
                      WHITE, cv2.FILLED)
        cv2.putText(im_res, text3,
                    (10, h_im // 2 + text_height + 10),
                    font, font_scale_normal, RED, thickness_normal)

        (text_width, text_height) = cv2.getTextSize(text4, font, fontScale=font_scale_normal, thickness=thickness_normal)[0]
        cv2.rectangle(im_res, (w_im // 2, h_im // 2),
                      (w_im // 2 + text_width + 30, h_im // 2 + text_height + 30),
                      WHITE, cv2.FILLED)
        cv2.putText(im_res, text4,
                    (w_im // 2 + 10, h_im // 2 + text_height + 10),
                    font, font_scale_normal, RED, thickness_normal)

    cv2.imshow('res', my_resize(im_res, height=900))


def show_thresh_adaptive(gray_enhance):
    while (1):
        cv2.imshow('image', gray_enhance)

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
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def show_trackbar_adap_thresh(gray_enhance):
    max_block = 40
    max_mean = 80
    cv2.namedWindow('track')
    cv2.createTrackbar('B1', 'track', block_size_big // 2 + 1, max_block, nothing)
    cv2.createTrackbar('M1', 'track', mean_sub_big, max_mean, nothing)
    cv2.createTrackbar('B', 'track', block_size_big // 2 + 1, max_block, nothing)
    cv2.createTrackbar('M', 'track', mean_sub_big, max_mean, nothing)
    cv2.createTrackbar('M/G', 'track', 0, 1, nothing)
    cv2.createTrackbar('blur_size', 'track', 1, 5, nothing)
    show_thresh_adaptive(gray_enhance)


def show_hough(edges):
    # cv2.imshow("edges", edges)
    # old_values = [-1,-1,-1]
    while (1):
        w = cv2.getTrackbarPos('width', 'track')
        edges_resize = my_resize(edges, width=max(100, w))
        cv2.imshow("edges_resize", edges_resize)

        A = cv2.getTrackbarPos('thresh', 'track')
        B = cv2.getTrackbarPos('minLineLength', 'track')
        C = cv2.getTrackbarPos('maxLineGa', 'track')
        rho = max(1, cv2.getTrackbarPos('rho', 'track'))
        theta = max(1, cv2.getTrackbarPos('theta', 'track')) * np.pi / 180
        my_lines = []

        img_lines = np.zeros((edges_resize.shape[:2]), np.uint8)

        lines_raw = cv2.HoughLinesP(edges_resize, rho=rho, theta=theta, threshold=A,
                                    minLineLength=B, maxLineGap=C)

        img_binary_lines = cv2.cvtColor(edges_resize, cv2.COLOR_GRAY2BGR)
        if lines_raw is not None:
            for line in lines_raw:
                my_lines.append(MyHoughPLines(line))

            for line in my_lines:
                x1, y1, x2, y2 = line.get_limits()
                cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)

            for line in my_lines:
                x1, y1, x2, y2 = line.get_limits()
                cv2.line(img_binary_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # cv2.imshow('img_lines', resize(img_lines, width=900))
        cv2.imshow('img_lines', img_lines)
        # cv2.imshow('img_binary_lines', resize(img_binary_lines, width=900))
        cv2.imshow('img_binary_lines', img_binary_lines)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break


def show_trackbar_hough(edges):
    max_thresh = 300
    max_maxLineGap = 20
    max_minLineLength = 20
    max_rho = 20
    max_theta = 20
    cv2.namedWindow('track')
    cv2.createTrackbar('width', 'track', 900, 1200, nothing)
    cv2.createTrackbar('thresh', 'track', 100, max_thresh, nothing)
    cv2.createTrackbar('minLineLength', 'track', 5, max_minLineLength, nothing)
    cv2.createTrackbar('maxLineGa', 'track', 5, max_maxLineGap, nothing)
    cv2.createTrackbar('rho', 'track', 1, max_rho, nothing)
    cv2.createTrackbar('theta', 'track', 1, max_theta, nothing)
    show_hough(edges)


def preprocess_im(frame, using_webcam=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    if display_prepro_big:
        show_trackbar_adap_thresh(gray_enhance)

    if using_webcam:
        thresh = cv2.adaptiveThreshold(gray_enhance, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       block_size_webcam_big, mean_sub_webcam_big)

        thresh_not = cv2.bitwise_not(thresh)

        dilate = cv2.morphologyEx(thresh_not, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))  # Delete space between line

        return dilate

        # kernel_close = np.ones((5, 5), np.uint8)
        # closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
        # return closing
    else:
        blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       block_size_big, mean_sub_big)

        thresh_not = cv2.bitwise_not(thresh)

        # kernel_open = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, kernel_open)  # Denoise
        kernel_close = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
        dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)  # Delete space between line
        #
        # return closing
        return dilate


def flood_fill_grid(thresh):
    max_area = -1
    maxPt = (0, 0)
    h_im, w_im = thresh.shape[:2]
    t_copy = thresh.copy()
    # grid = thresh.copy()
    # mask = np.zeros((h_im + 2, w_im + 2), np.uint8)
    for y in range(2, h_im - 2):
        for x in range(2, w_im - 2):
            if thresh[y][x] > 128:
                # ret = cv2.floodFill(t_copy, mask, (x,y), 64)
                ret = cv2.floodFill(t_copy, None, (x, y), 1)

                area = ret[0]
                if area > max_area:
                    max_area = area
                    maxPt = (x, y)

    cv2.floodFill(t_copy, None, maxPt, 255)
    return t_copy


def line_intersection(my_line1, my_line2):
    line1 = [[my_line1[0], my_line1[1]], [my_line1[2], my_line1[3]]]
    line2 = [[my_line2[0], my_line2[1]], [my_line2[2], my_line2[3]]]
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [int(x), int(y)]


def look_for_intersections_hough(lines):
    hor_up = (1000, 1000, 1000, 1000)  # x1,y1,x2,y2
    hor_down = (0, 0, 0, 0)  # x1,y1,x2,y2
    ver_left = (1000, 1000, 1000, 1000)  # x1,y1,x2,y2
    ver_right = (0, 0, 0, 0)  # x1,y1,x2,y2

    for line in [line for line in lines if not line.isMerged]:
        lim = line.get_limits()
        if line.theta < np.pi / 4:  # Ligne Verticale
            if lim[0] + lim[2] < ver_left[0] + ver_left[2]:
                ver_left = lim
            elif lim[0] + lim[2] > ver_right[0] + ver_right[2]:
                ver_right = lim
        else:
            if lim[1] + lim[3] < hor_up[1] + hor_up[3]:
                hor_up = lim
            elif lim[1] + lim[3] > hor_down[1] + hor_down[3]:
                hor_down = lim
    # raw_limits_lines = [hor_up, hor_down, ver_left, ver_right]

    grid_limits = list()
    grid_limits.append(line_intersection(hor_up, ver_left))
    grid_limits.append(line_intersection(hor_up, ver_right))
    grid_limits.append(line_intersection(hor_down, ver_right))
    grid_limits.append(line_intersection(hor_down, ver_left))

    return grid_limits


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    # contour_x = sorted(contour,key = lambda c:c[0][0])
    # contour_y = sorted(contour,key = lambda c:c[0][1])
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])

    for j in range(len(contour)):
        x, y = contour[j][0]
        if x > mean_x:  # On right
            if y > mean_y:  # On bottom
                bottom_right = [x, y]
            else:
                top_right = [x, y]
        else:
            if y > mean_y:  # On bottom
                bottom_left = [x, y]
            else:
                top_left = [x, y]
    return [top_left, top_right, bottom_right, bottom_left]


def look_for_corners(img_lines, display=False):
    if display:
        # img_contours = np.zeros((img_lines.shape[0], img_lines.shape[1], 3), np.uint8)
        img_contours = cv2.cvtColor(img_lines.copy(), cv2.COLOR_GRAY2BGR)
    else:
        img_contours = None

    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    biggest_area = cv2.contourArea(contours[0])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < smallest_area_allow:
            break
        if area > biggest_area / ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_poly_coef * peri, True)
            if len(approx) == 4:
                best_contours.append(approx)
        if display:
            cv2.drawContours(img_contours, [cnt], 0, (0, 255, 0), 2)

    if not best_contours:
        if not display:
            return None
        else:
            return None, img_lines, img_contours
    corners = []
    for best_contour in best_contours:
        corners.append(find_corners(best_contour))

    if not display:
        return corners
    else:
        for best_contour in best_contours:
            cv2.drawContours(img_contours, [best_contour], 0, (0, 0, 255), 3)
            for corner in corners:
                for point in corner:
                    x, y = point
                    cv2.circle(img_contours, (x, y), 10, (255, 0, 0), 3)
        return corners, img_lines, img_contours


# @timer_decorator
def get_lines_and_corners(img, edges, use_hough=False, display=False):
    if use_hough:
        # t0 = time.time()
        my_lines = []
        img_lines = np.zeros((img.shape[:2]), np.uint8)
        # edges_resize = my_resize(edges, width=resize_width_hough,height=resize_height_hough)
        # print(edges_resize.shape)
        lines_raw = cv2.HoughLinesP(edges,
                                    rho=hough_rho, theta=hough_theta,
                                    threshold=thresh_hough_p,
                                    minLineLength=minLineLength_h_p, maxLineGap=maxLineGap_h_p)
        # t1 = time.time()

        for line in lines_raw:
            # my_lines.append(MyHoughPLines(line, ratio=ratio_resize_hough))
            my_lines.append(MyHoughPLines(line))
        # t2 = time.time()

        for line in my_lines:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)
        # cv2.imshow('img_lines', img_lines)
        # cv2.waitKey()
        if display_line_on_edges:
            show_trackbar_hough(edges)
        # t3 = time.time()
        #
        # total_time = t3 - t0
        # prepro_time = t1 - t0
        # print("INSIDE Hough Transfrom \t{:.1f}% - {:.3f}s".format(100 * prepro_time / total_time, prepro_time))
        # hough_time = t2 - t1
        # print("INSIDE LINES \t\t{:.1f}% - {:.3f}s".format(100 * hough_time / total_time, hough_time))
        # undistort_time = t3 - t2
        # print("INSIDE DRAWS LINE \t{:.1f}% - {:.3f}s".format(100 * undistort_time / total_time, undistort_time))
        # print("INSIDE EVERYTHING DONE \t{:.2f}s".format(total_time))

    else:
        img_lines = edges.copy()
    return look_for_corners(img_lines, display)


def get_hough_transform(img, edges, display=False):
    my_lines = []
    img_after_merge = img.copy()
    lines_raw = cv2.HoughLines(edges, 1, np.pi / 180, thresh_hough)
    for line in lines_raw:
        my_lines.append(MyHoughLines(line))

    if display:
        for line in my_lines:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    merge_lines(my_lines)
    grid_limits = look_for_intersections_hough(my_lines)

    if display:
        for line in [line for line in my_lines if not line.isMerged]:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img_after_merge, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for point in grid_limits:
        x, y = point
        cv2.circle(img_after_merge, (x, y), 10, (255, 0, 0), 3)

    if not display:
        return grid_limits
    else:
        return grid_limits, img, img_after_merge

import time
def undistorted_grids(frame, points_grids, ratio):
    undistorted = []
    true_points_grids = []
    transfo_matrix = []
    for points_grid in points_grids:
        points_grid = np.array(points_grid, dtype=np.float32) * ratio
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        M = cv2.getPerspectiveTransform(points_grid, final_pts)
        undistorted.append(cv2.warpPerspective(frame, M, (target_w_grid, target_h_grid)))
        # cv2.imshow("test",undistorted[-1])
        # cv2.waitKey()
        true_points_grids.append(points_grid)
        transfo_matrix.append(np.linalg.inv(M))
    return undistorted, true_points_grids, transfo_matrix


def main_grid_detector_img(frame, resized=True, display=False, using_webcam=False, use_hough=False):
    if not resized:
        frame_resize = my_resize(frame, width=param_resize_width, height=param_resize_height)
    else:
        frame_resize = frame
    ratio = frame.shape[0] / frame_resize.shape[0]
    prepro_im_edges = preprocess_im(frame_resize, using_webcam)

    if display:
        extreme_points_biased, img_lines, img_contour = get_lines_and_corners(frame_resize.copy(), prepro_im_edges,
                                                                              use_hough=use_hough, display=display)
        show_big_image(frame_resize, prepro_im_edges, img_lines, img_contour, use_hough)

    else:
        extreme_points_biased = get_lines_and_corners(frame_resize.copy(), prepro_im_edges, use_hough=use_hough,
                                                      display=display)

    if extreme_points_biased is None:
        return None, None, None
    grids_final, points_grids, transfo_matrix = undistorted_grids(frame, extreme_points_biased, ratio)
    return grids_final, points_grids, transfo_matrix


if __name__ == '__main__':
    # im_path = "dataset_test/021.jpg"
    im_path = "images_test/sudoku.jpg"
    # im_path = "tmp/030.jpg"
    # im_path = "images_test/imagedouble.jpg"
    # im_path = "images_test/izi_distord.jpg"
    im = cv2.imread(im_path)
    cv2.imshow("im",im)
    res_grids_final, _, _ = main_grid_detector_img(im, resized=False, display=True,
                                                   using_webcam=False, use_hough=True)
    if res_grids_final is not None:
        for (i, im_grid) in enumerate(res_grids_final):
            cv2.imshow('grid_final_{}'.format(i), im_grid)
            cv2.imwrite('images_test/grid_cut_{}.jpg'.format(i), im_grid)
    cv2.waitKey()
