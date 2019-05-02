import cv2
from src.MyHoughLines import *
from src.MyHoughPLines import *
import imutils


def preprocess_im(gray_im):
    blurred = cv2.GaussianBlur(gray_im, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_not = cv2.bitwise_not(thresh)
    kernel_open = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, kernel_open)  # Denoise
    kernel_close = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
    # dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel_close) # Delete space between line

    return closing
    # return thresh_not


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


# size_min_contours_ratio = 1.0/30
size_min_contours = 100 * 2
ratio_lim = 1.5


def look_for_corners(img_lines, display=False):
    # size_min_contours = size_min_contours_ratio * img_lines.shape[0] * img_lines.shape[1]
    if display:
        img_contours = np.zeros((img_lines.shape[0], img_lines.shape[1], 3), np.uint8)
    else:
        img_contours = None

    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    best_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > biggest_area * ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
            if len(approx) == 4:
                best_contours = [approx]
                biggest_area = area
        elif area > biggest_area / ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
            if len(approx) == 4:
                best_contours.append(approx)
        if display and area > size_min_contours:
            cv2.drawContours(img_contours, [cnt], 0, (0, 255, 0), 2)

    if not best_contours:
        if not display:
            return None
        else:
            return None, img_lines, img_contours
    corners = []
    for best_contour in best_contours:
        extLeft = best_contour[best_contour[:, :, 0].argmin()][0]
        extRight = best_contour[best_contour[:, :, 0].argmax()][0]
        extTop = best_contour[best_contour[:, :, 1].argmin()][0]
        extBot = best_contour[best_contour[:, :, 1].argmax()][0]

        if extLeft[1] < extRight[1]:
            top_left = extLeft
            top_right = extTop
            bottom_right = extRight
            bottom_left = extBot
        else:
            top_left = extTop
            top_right = extRight
            bottom_right = extBot
            bottom_left = extLeft
        # cv2.waitKey()
        corners.append([top_left, top_right, bottom_right, bottom_left])
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


thresh_hough = 500
thresh_hough_p = 100
minLineLength_h_p = 0
maxLineGap_h_p = 3


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


def get_p_hough_transform(img, edges, display=False):
    my_lines = []
    img_binary_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_lines = np.zeros((img.shape[:2]), np.uint8)
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh_hough_p, minLineLength_h_p, maxLineGap_h_p)
    for line in lines_raw:
        my_lines.append(MyHoughPLines(line))

    for line in my_lines:
        x1, y1, x2, y2 = line.get_limits()
        cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)
    # cv2.imshow('img_lines', img_lines)
    # cv2.waitKey()
    if display:
        for line in my_lines:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img_binary_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return look_for_corners(img_lines, display)


target_h, target_w = 450, 450


def undistorted_grids(im, points_grids, ratio):
    undistorted = []
    true_points_grids = []
    for points_grid in points_grids:
        points_grid = np.array(points_grid, dtype=np.float32)
        # h_im, w_im = im.shape[:2]
        # final_pts = np.array([[0, 0], [h_im - 1, 0], [h_im - 1, w_im - 1], [0, w_im - 1]], dtype=np.float32)
        final_pts = np.array([[0, 0], [target_h - 1, 0], [target_h - 1, target_w - 1], [0, target_w - 1]],
                             dtype=np.float32)
        M = cv2.getPerspectiveTransform(points_grid, final_pts)
        undistorted.append(cv2.warpPerspective(im, M, (target_w, target_h)))
        true_points_grids.append(points_grid * ratio)
    return undistorted, true_points_grids


def look_for_new_grid(frame):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame, 50, 150)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('canny', canny)


def main_grid_detector_video(display=False):
    iso_grid = False
    grid = None
    cap = cv2.VideoCapture(1)

    while "user does not exit":
        # Capture frame-by-frame
        _, frame = cap.read()

        look_for_new_grid(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ord('q')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main_grid_detector_img(frame, display=False, resize=False):
    # flood_fill_grid = False
    # grid = None
    use_p_hough = True
    if not resize:
        frame_resize = imutils.resize(frame, height=900, width=900)
    else:
        frame_resize = frame
    ratio = frame.shape[0] / frame_resize.shape[0]

    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    prepro_im = preprocess_im(gray)  # Good old OTSU
    # if flood_fill_grid:
    #     grid = flood_fill_grid(prepro_im)
    # canny = cv2.Canny(frame, 50, 150)

    if display:
        if use_p_hough:
            extreme_points, img_lines, img_contour = get_p_hough_transform(frame_resize.copy(), prepro_im, display)
            cv2.imshow('prepro_im', prepro_im)
            cv2.imshow('img_lines', img_lines)
            cv2.imshow('img_contour', img_contour)
        else:
            extreme_points, img_hough, img_after_merge = get_hough_transform(frame_resize.copy(), prepro_im, display)
            cv2.imshow('prepro_im', prepro_im)
            cv2.imshow('img_hough', img_hough)
            cv2.imshow('img_after_merge', img_after_merge)
    else:
        if use_p_hough:
            extreme_points = get_p_hough_transform(frame_resize.copy(), prepro_im)
        else:
            extreme_points = get_hough_transform(frame_resize.copy(), prepro_im)

    if extreme_points is None:
        return None, None

    grids_final, points_grids = undistorted_grids(frame_resize, extreme_points, ratio)
    return grids_final, points_grids


if __name__ == '__main__':
    # im_path = "../images/sudoku2.jpg"
    im_path = "../images/imagedouble.jpg"
    im = cv2.imread(im_path)
    res_grids_final, _ = main_grid_detector_img(im, True)
    if res_grids_final is not None:
        for (i, im_grid) in enumerate(res_grids_final):
            cv2.imshow('grid_final_{}'.format(i), im_grid)
            cv2.imwrite('../images/grid_cut_{}.jpg'.format(i), im_grid)
    cv2.waitKey()
