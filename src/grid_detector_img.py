import cv2
from src.MyHoughLines import *
from src.MyHoughPLines import *
# import imutils
from src.fonctions import resize


def nothing(x):
    pass


def show_thresh_adaptive(gray_enhance):
    while (1):
        cv2.imshow('image', gray_enhance)

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
        blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(gray_enhance, 255, adap, cv2.THRESH_BINARY, A, B)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, adap, cv2.THRESH_BINARY, C, D)

        cv2.imshow('thresh', thresh)
        cv2.imshow('thresh2', thresh2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


def show_hough(edges):
    cv2.imshow("edges", resize(edges, width=900))
    # old_values = [-1,-1,-1]
    while (1):
        # get current positions of four trackbars
        A = cv2.getTrackbarPos('thresh', 'track')
        B = cv2.getTrackbarPos('minLineLength', 'track')
        C = cv2.getTrackbarPos('maxLineGa', 'track')
        my_lines = []
        img_lines = np.zeros((edges.shape[:2]), np.uint8)
        lines_raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=A,
                                    minLineLength=B, maxLineGap=C)

        for line in lines_raw:
            my_lines.append(MyHoughPLines(line))

        for line in my_lines:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)

        img_binary_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for line in my_lines:
            x1, y1, x2, y2 = line.get_limits()
            cv2.line(img_binary_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('img_lines', resize(img_lines, width=900))
        cv2.imshow('img_binary_lines', resize(img_binary_lines, width=900))
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break


def show_trackbar_adap_thresh(gray_enhance):
    max_block = 40
    max_mean = 20
    cv2.namedWindow('track')
    cv2.createTrackbar('B1', 'track', 10, max_block, nothing)
    cv2.createTrackbar('M1', 'track', 13, max_mean, nothing)
    cv2.createTrackbar('B', 'track', 10, max_block, nothing)
    cv2.createTrackbar('M', 'track', 13, max_mean, nothing)
    cv2.createTrackbar('M/G', 'track', 0, 1, nothing)
    show_thresh_adaptive(gray_enhance)


def show_trackbar_hough(edges):
    max_thresh = 300
    max_maxLineGap = 20
    max_minLineLength = 20
    cv2.namedWindow('track')
    cv2.createTrackbar('thresh', 'track', 100, max_thresh, nothing)
    cv2.createTrackbar('minLineLength', 'track', 0, max_minLineLength, nothing)
    cv2.createTrackbar('maxLineGa', 'track', 4, max_maxLineGap, nothing)
    show_hough(edges)


def preprocess_im(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    # show_trackbar_adap_thresh(gray_enhance)
    blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7)
    thresh_not = cv2.bitwise_not(thresh)

    # kernel_open = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, kernel_open)  # Denoise
    kernel_close = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
    # dilate = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel_close) # Delete space between line
    #
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


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    # contour_x = sorted(contour,key = lambda c:c[0][0])
    # contour_y = sorted(contour,key = lambda c:c[0][1])
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])

    for i in range(len(contour)):
        x, y = contour[i][0]
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


# size_min_contours_ratio = 1.0/30
ratio_lim = 1.5
smallest_area = 75000
approx_poly_coef = 0.1


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
        if area < smallest_area:
            continue
        if area > biggest_area * ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_poly_coef * peri, True)
            if len(approx) == 4:
                best_contours = [approx]
                biggest_area = area
        elif area > biggest_area / ratio_lim:
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


thresh_hough = 500
thresh_hough_p = 100
minLineLength_h_p = 5
maxLineGap_h_p = 5
# display_line_on_edges = True
display_line_on_edges = False


def get_p_hough_transform(img, edges, display=False):
    my_lines = []
    img_lines = np.zeros((img.shape[:2]), np.uint8)
    lines_raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=thresh_hough_p,
                                minLineLength=minLineLength_h_p, maxLineGap=maxLineGap_h_p)
    for line in lines_raw:
        my_lines.append(MyHoughPLines(line))

    for line in my_lines:
        x1, y1, x2, y2 = line.get_limits()
        cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2)
    # cv2.imshow('img_lines', img_lines)
    # cv2.waitKey()
    if display_line_on_edges:
        show_trackbar_hough(edges)
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
        # cv2.imshow("test",undistorted[-1])
        # cv2.waitKey()
        true_points_grids.append(points_grid * ratio)
    return undistorted, true_points_grids


def main_grid_detector_img(frame, display=False, resized=False):
    # flood_fill_grid = False
    # grid = None
    use_p_hough = True
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not resized:
        frame_resize = resize(frame, height=900, width=900)
    else:
        frame_resize = frame
    ratio = frame.shape[0] / frame_resize.shape[0]
    prepro_im = preprocess_im(frame_resize)
    # if flood_fill_grid:
    #     grid = flood_fill_grid(prepro_im)
    # canny = cv2.Canny(frame, 50, 150)

    if display:
        if use_p_hough:
            extreme_points, img_lines, img_contour = get_p_hough_transform(frame_resize.copy(), prepro_im, display)
            cv2.imshow('frame_resize', frame_resize)
            cv2.imshow('prepro_im', prepro_im)
            cv2.imshow('img_lines', img_lines)
            cv2.imshow('img_contour', img_contour)
        else:
            extreme_points, img_hough, img_after_merge = get_hough_transform(frame_resize.copy(), prepro_im, display)
            cv2.imshow('frame_resize', frame_resize)
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
    # im_path = "../dataset_test/115.jpg"
    # im_path = "../images_test/sudoku5.jpg"
    # im_path = "../images_test/sudoku6.jpg"
    im_path = "../images_test/video_stop.png"
    # im_path = "../images_test/imagedouble.jpg"
    im = cv2.imread(im_path)

    res_grids_final, _ = main_grid_detector_img(im, True)
    if res_grids_final is not None:
        for (i, im_grid) in enumerate(res_grids_final):
            cv2.imshow('grid_final_{}'.format(i), im_grid)
            # cv2.imwrite('../images_test/grid_cut_{}.jpg'.format(i), im_grid)
    cv2.waitKey()
