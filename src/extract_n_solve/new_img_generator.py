import cv2
import numpy as np


class ImageGenerator:
    def __init__(self):
        self.__color = (0, 155, 255)

        self.__font = cv2.FONT_HERSHEY_SIMPLEX
        self.__font_scale = 1.2
        self.__thickness = 2

    def create_image_filled(self, img, unwraped_grid_list, grids_matrix, grids_solved,
                            points_grids, list_transform_matrix):
        ims_filled_grid = self.write_solved_grids(unwraped_grid_list, grids_matrix, grids_solved)
        im_final = self.recreate_img_filled(img, ims_filled_grid, points_grids, list_transform_matrix)

        return im_final

    def write_solved_grids(self, frames, grids_matrix, solved_grids):
        ims_filled_grid = []
        for frame, grid_init, solved_grid in zip(frames, grids_matrix, solved_grids):
            if solved_grid is None:
                ims_filled_grid.append(None)
                continue
            im_filled_grid = np.zeros_like(frame)
            h_im, w_im = frame.shape[:2]
            for y in range(9):
                for x in range(9):
                    if grid_init[y, x] != 0 or solved_grid[y, x] == 0:
                        continue
                    true_y, true_x = int((y + 0.5) * h_im / 9), int((x + 0.5) * w_im / 9)
                    digit = str(solved_grid[y, x])
                    (text_width, text_height) = cv2.getTextSize(digit, self.__font, fontScale=self.__font_scale,
                                                                thickness=self.__thickness)[0]
                    cv2.putText(im_filled_grid, digit,
                                (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                                self.__font, self.__font_scale, (0, 3, 0), self.__thickness * 3)
                    cv2.putText(im_filled_grid, digit,
                                (true_x - int(text_width / 2), true_y + int(text_height / 2)),
                                self.__font, self.__font_scale, (0, 255, 0), self.__thickness)
            ims_filled_grid.append(im_filled_grid)
        return ims_filled_grid

    @staticmethod
    def recreate_img_filled(frame, im_grids, points_grids, list_transform_matrix):
        target_h, target_w = frame.shape[:2]
        new_im = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        for im_grid, points_grid, transform_matrix in zip(im_grids, points_grids, list_transform_matrix):
            if im_grid is None:
                for point in points_grid:
                    x, y = point
                    cv2.circle(new_im, (x, y), 6, (255, 0, 0), 3)

            else:
                new_im = cv2.add(new_im, cv2.warpPerspective(im_grid, transform_matrix, (target_w, target_h)))

        _, mask = cv2.threshold(cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        im_final = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        im_final = cv2.add(im_final, new_im)

        return im_final


if __name__ == '__main__':
    im_path = "images_test/grid_cut_0.jpg"
    im = cv2.imread(im_path)
    im = cv2.resize(im, (450, 450))
    my_grid_init = [[9, 0, 0, 3, 4, 0, 0, 0, 0],
                    [0, 0, 0, 9, 0, 0, 7, 0, 0],
                    [2, 0, 0, 0, 1, 0, 8, 0, 0],
                    [0, 0, 0, 6, 0, 0, 2, 7, 0],
                    [0, 3, 0, 0, 2, 0, 0, 1, 0],
                    [0, 5, 2, 0, 0, 9, 0, 0, 0],
                    [0, 0, 8, 0, 6, 0, 0, 0, 5],
                    [0, 0, 0, 0, 9, 1, 0, 0, 4],
                    [0, 0, 4, 0, 0, 8, 0, 0, 0]]

    my_solved_grid = [[5, 6, 3, 9, 8, 2, 7, 4, 1],
                      [9, 8, 1, 3, 4, 7, 6, 5, 2],
                      [2, 4, 7, 5, 1, 6, 8, 9, 3],
                      [4, 1, 9, 6, 3, 5, 2, 7, 8],
                      [7, 3, 6, 8, 2, 4, 5, 1, 9],
                      [8, 5, 2, 1, 7, 9, 4, 3, 6],
                      [1, 7, 8, 4, 6, 3, 9, 2, 5],
                      [6, 2, 5, 7, 9, 1, 3, 8, 4],
                      [3, 9, 4, 2, 5, 8, 1, 6, 7]]

    generator = ImageGenerator()
    res_im_filled_grid = generator.write_solved_grids([im], [np.array(my_grid_init)], np.array([my_solved_grid]))
    cv2.imshow("im", im)
    cv2.imshow("im_fill", res_im_filled_grid[0])
    cv2.waitKey()

