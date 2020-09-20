import cv2

from settings import *
from src.solving_objects.MyHoughLines import *
from src.solving_objects.MyHoughPLines import *


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
    detector = GridDetector()
    res_grids_final, _, _ = detector.extract_grids(im)
    if res_grids_final is not None:
        for (i, im_grid) in enumerate(res_grids_final):
            cv2.imshow('grid_final_{}'.format(i), im_grid)
            cv2.imwrite('images_test/grid_cut_{}.jpg'.format(i), im_grid)
    cv2.waitKey()
