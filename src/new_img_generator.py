import numpy as np
import cv2

color = (0, 155, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 2


def write_FPS(im, elapsed_time):
    cv2.putText(im, "{:.2f} FPS".format(1 / elapsed_time),
                (50, 80), font, 3, (0, 255, 0), thickness)


def recreate_img_filled(frame, im_grids, points_grids):
    target_h, target_w = frame.shape[:2]
    im_final = frame.copy()
    for im_grid, points_grid in zip(im_grids, points_grids):
        if im_grid is None:
            new_im = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
            for point in points_grid:
                x, y = point
                cv2.circle(new_im, (x, y), 6, (255, 0, 0), 3)
            _, mask = cv2.threshold(cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

        else:
            grid_h, grid_w = im_grid.shape[:2]
            init_pts = np.array([[0, 0], [grid_h - 1, 0], [grid_h - 1, grid_w - 1], [0, grid_w - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(init_pts, points_grid)
            new_im = cv2.warpPerspective(im_grid, M, (target_w, target_h))
            _, mask = cv2.threshold(cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        im_final = cv2.bitwise_and(im_final, im_final, mask=cv2.bitwise_not(mask))
        im_final = cv2.add(im_final, new_im)
    return im_final


def write_solved_grids(frames, grids_matrix, solved_grids):
    ims_filled_grid = []
    for frame, grid_init, solved_grid in zip(frames, grids_matrix, solved_grids):
        if solved_grid is None:
            ims_filled_grid.append(None)
            continue
        im_filled_grid = np.zeros_like(frame)
        h_im, w_im = frame.shape[:2]
        for y in range(9):
            for x in range(9):
                if grid_init[y, x] != 0:
                    continue
                true_y, true_x = int((y + 0.5) * h_im / 9), int((x + 0.5) * w_im / 9)
                digit = str(solved_grid[y, x])
                (text_width, text_height) = cv2.getTextSize(digit, font, fontScale=font_scale, thickness=thickness)[0]
                cv2.putText(im_filled_grid, digit,
                            (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                            font, font_scale, (0, 3, 0), thickness * 3)
                cv2.putText(im_filled_grid, digit,
                            (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                            font, font_scale, (0, 255, 0), thickness)
        ims_filled_grid.append(im_filled_grid)
    return ims_filled_grid


if __name__ == '__main__':
    im_path = "../images_test/grid_cut_3.jpg"
    img = cv2.imread(im_path)
    img = cv2.resize(img, (450, 450))
    my_grid_init = [[9, 0, 0, 3, 4, 0, 0, 0, 0]
        , [0, 0, 0, 9, 0, 0, 7, 0, 0]
        , [2, 0, 0, 0, 1, 0, 8, 0, 0]
        , [0, 0, 0, 6, 0, 0, 2, 7, 0]
        , [0, 3, 0, 0, 2, 0, 0, 1, 0]
        , [0, 5, 2, 0, 0, 9, 0, 0, 0]
        , [0, 0, 8, 0, 6, 0, 0, 0, 5]
        , [0, 0, 0, 0, 9, 1, 0, 0, 4]
        , [0, 0, 4, 0, 0, 8, 0, 0, 0]]

    my_solved_grid = [[5, 6, 3, 9, 8, 2, 7, 4, 1]
        , [9, 8, 1, 3, 4, 7, 6, 5, 2]
        , [2, 4, 7, 5, 1, 6, 8, 9, 3]
        , [4, 1, 9, 6, 3, 5, 2, 7, 8]
        , [7, 3, 6, 8, 2, 4, 5, 1, 9]
        , [8, 5, 2, 1, 7, 9, 4, 3, 6]
        , [1, 7, 8, 4, 6, 3, 9, 2, 5]
        , [6, 2, 5, 7, 9, 1, 3, 8, 4]
        , [3, 9, 4, 2, 5, 8, 1, 6, 7]]
    res_im_filled_grid = write_solved_grids([img], [np.array(my_grid_init)], np.array([my_solved_grid]))
    cv2.imshow("im", img)
    cv2.imshow("im_fill", res_im_filled_grid[0])
    cv2.waitKey()
