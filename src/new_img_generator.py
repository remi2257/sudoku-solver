import numpy as np
import cv2

color = (0, 155, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
thickness = 3


def recreate_img(im_init, im_grid, M):
    h_im, w_im = im_init.shape[:2]
    M_inv = np.linalg.inv(M)
    new_im = cv2.warpPerspective(im_grid, M_inv, (w_im, h_im))
    _, mask = cv2.threshold(cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    init_cut = cv2.bitwise_and(im_init, im_init, mask=cv2.bitwise_not(mask))
    im_final = cv2.add(init_cut, new_im)

    return im_final


def write_solved_grid(im_grid, grid_init, solved_grid):
    # im_filled_grid = im_grid.copy()
    im_filled_grid = np.zeros_like(im_grid)
    h_im, w_im = im_grid.shape[:2]

    for y in range(9):
        for x in range(9):
            if grid_init[y, x] != 0:
                continue
            true_y, true_x = int((y + 0.5) * h_im / 9), int((x + 0.5) * w_im / 9)
            digit = str(solved_grid[y, x])
            (text_width, text_height) = cv2.getTextSize(digit, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale, (0, 3, 0), thickness*3)
            cv2.putText(im_filled_grid, digit,
                        (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                        font, font_scale, (0, 255, 0), thickness)
    return im_filled_grid


if __name__ == '__main__':
    im_path = "images/grid_cut1.jpg"
    img = cv2.imread(im_path)
    grid_init = [[0, 0, 0, 9, 0, 0, 7, 0, 0]
        , [9, 0, 0, 3, 4, 0, 0, 0, 0]
        , [2, 0, 0, 0, 1, 0, 8, 0, 0]
        , [0, 0, 0, 6, 0, 0, 2, 7, 0]
        , [0, 3, 0, 0, 2, 0, 0, 1, 0]
        , [0, 5, 2, 0, 0, 9, 0, 0, 0]
        , [0, 0, 8, 0, 6, 0, 0, 0, 5]
        , [0, 0, 0, 0, 9, 1, 0, 0, 4]
        , [0, 0, 4, 0, 0, 8, 0, 0, 0]]

    solved_grid = [[5, 6, 3, 9, 8, 2, 7, 4, 1]
        ,[9, 8, 1, 3, 4, 7, 6, 5, 2]
        , [2, 4, 7, 5, 1, 6, 8, 9, 3]
        , [4, 1, 9, 6, 3, 5, 2, 7, 8]
        , [7, 3, 6, 8, 2, 4, 5, 1, 9]
        , [8, 5, 2, 1, 7, 9, 4, 3, 6]
        , [1, 7, 8, 4, 6, 3, 9, 2, 5]
        , [6, 2, 5, 7, 9, 1, 3, 8, 4]
        , [3, 9, 4, 2, 5, 8, 1, 6, 7]]
    write_solved_grid(img, np.array(grid_init), np.array(solved_grid))
    cv2.imshow("im", img)
    cv2.waitKey()
