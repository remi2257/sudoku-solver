import cv2
from src.MyHoughLines import *


def preprocess_im(gray_im):
    blurred = cv2.GaussianBlur(gray_im, (5, 5), 0)

    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2);
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_not = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    # closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, kernel)

    return opening


def isolate_grid(thresh):
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


def look_for_intersections(lines):
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


def get_hough_transform_display(img, edges):
    lines_raw = cv2.HoughLines(edges, 1, np.pi / 180, 500)
    my_lines = []
    img_after_merge = img.copy()

    for line in lines_raw:
        my_lines.append(MyHoughLines(line))

    for line in my_lines:
        x1, y1, x2, y2 = line.get_limits()
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    merge_lines(my_lines)

    for line in [line for line in my_lines if not line.isMerged]:
        x1, y1, x2, y2 = line.get_limits()
        cv2.line(img_after_merge, (x1, y1), (x2, y2), (255, 0, 0), 2)

    grid_limits = look_for_intersections(my_lines)

    for point in grid_limits:
        x, y = point
        cv2.circle(img_after_merge, (x, y), 10, (0, 255, 0), 3)

    return grid_limits, img, img_after_merge


def get_hough_transform(img, edges):
    lines_raw = cv2.HoughLines(edges, 1, np.pi / 180, 500)
    my_lines = []
    img_after_merge = img.copy()

    for line in lines_raw:
        my_lines.append(MyHoughLines(line))

    merge_lines(my_lines)

    grid_limits = look_for_intersections(my_lines)

    return grid_limits


def undistorted_grid(im, points_grid):
    h_im, w_im = im.shape[:2]
    final_pts = np.array([[0, 0], [h_im - 1, 0], [h_im - 1, w_im - 1], [0, w_im - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(points_grid, dtype=np.float32), final_pts)
    undistorted = cv2.warpPerspective(im, M, (h_im, w_im))

    return undistorted, M


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


def main_grid_detector_img(frame, display=False):
    iso_grid = False
    grid = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prepro_im = preprocess_im(gray)  # Good old OTSU
    if iso_grid:
        grid = isolate_grid(prepro_im)
    # canny = cv2.Canny(frame, 50, 150)

    if display:
        extreme_points, img_hough, img_after_merge = get_hough_transform_display(frame.copy(), prepro_im)
        cv2.imshow('prepro_im', prepro_im)
        if iso_grid:
            cv2.imshow('grid', grid)
        cv2.imshow('img_hough', img_hough)
        cv2.imshow('img_after_merge', img_after_merge)
    else:
        extreme_points = get_hough_transform(frame.copy(), prepro_im)

    grid_final, M = undistorted_grid(frame, extreme_points)

    return grid_final , M


if __name__ == '__main__':
    im_path = "images/sudoku.jpg"
    im = cv2.imread(im_path)
    res_grid_final, _ = main_grid_detector_img(im)
    cv2.imshow('grid_final', res_grid_final)
    # cv2.imwrite('images/grid_cut1.jpg',grid_final)
    cv2.waitKey()
