import cv2

from settings import *
from src.solving_objects.MyHoughLines import *
from src.solving_objects.MyHoughPLines import *
from src.useful_functions import my_resize


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


def get_undistorted_grids(frame, points_grids, ratio):
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
    grids_final, points_grids, transfo_matrix = get_undistorted_grids(frame, extreme_points_biased, ratio)
    return grids_final, points_grids, transfo_matrix


class GridDetector:
    def __init__(self, display=False):
        self.__display = display

    def extract_grids(self, frame):
        # Get a threshed image which emphasize lines
        threshed_img = self.thresh_img(frame)

        # Look for grids corners
        grids_corners_list = self.look_for_grids_corners(threshed_img)

        # Use grids corners to unwrap img !
        unwraped_grid_list, transfo_matrix = self.unwrap_grids(frame, grids_corners_list)
        return unwraped_grid_list, grids_corners_list, transfo_matrix

    @staticmethod
    def thresh_img(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))

        blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       block_size_big, mean_sub_big)

        thresh_not = cv2.bitwise_not(thresh)

        kernel_close = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
        dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)  # Delete space between line
        return dilate

    @staticmethod
    def unwrap_grids(frame, points_grids):
        undistorted_grids = []
        transfo_matrix_list = []
        for points_grid in points_grids:
            final_pts = np.array(
                [[0, 0], [target_w_grid - 1, 0],
                 [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
                dtype=np.float32)
            transfo_mat = cv2.getPerspectiveTransform(points_grid, final_pts)
            undistorted_grids.append(cv2.warpPerspective(frame, transfo_mat, (target_w_grid, target_h_grid)))
            transfo_matrix_list.append(np.linalg.inv(transfo_mat))
        return undistorted_grids, transfo_matrix_list

    @staticmethod
    def look_for_grids_corners(img_lines):
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

        if not best_contours:
            return None
        corners = []
        for best_contour in best_contours:
            corners.append(find_corners(best_contour))

        return np.array(corners, dtype=np.float32)


if __name__ == '__main__':
    # im_path = "dataset_test/021.jpg"
    im_path = "images_test/sudoku.jpg"
    # im_path = "tmp/030.jpg"
    # im_path = "images_test/imagedouble.jpg"
    # im_path = "images_test/izi_distord.jpg"
    im = cv2.imread(im_path)
    cv2.imshow("im", im)
    res_grids_final, _, _ = main_grid_detector_img(im, resized=False, display=True,
                                                   using_webcam=False, use_hough=True)
    if res_grids_final is not None:
        for (i, im_grid) in enumerate(res_grids_final):
            cv2.imshow('grid_final_{}'.format(i), im_grid)
            cv2.imwrite('images_test/grid_cut_{}.jpg'.format(i), im_grid)
    cv2.waitKey()
