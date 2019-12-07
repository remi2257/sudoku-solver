import numpy as np

from settings import same_grid_dist_ratio


class SudokuVideo:

    def __init__(self, grid):
        self.grid_raw = grid
        self.grid = np.zeros((9, 9), dtype=int)
        self.init_grid(grid)
        self.grid_solved = np.zeros((9, 9), dtype=int)
        self.isConfident = False
        self.isSolved = False
        self.nbr_apparition = 1
        self.last_apparition = 0
        self.TL = 0
        self.TR = 0
        self.BR = 0
        self.BL = 0
        self.w = 0
        self.h = 0

    def get_limits(self):
        return self.TL, self.TR, self.BR, self.BL

    def set_limits(self, points):
        self.TL = points[0]
        self.TR = points[1]
        self.BR = points[2]
        self.BL = points[3]

        self.w = ((self.TR[0] - self.TL[0]) + (self.BR[0] - self.BL[0])) / 2
        self.h = ((self.TR[1] - self.TL[1]) + (self.BR[1] - self.BL[1])) / 2

    def __str__(self):
        string = "-" * 18
        for y in range(9):
            string += "\n|"
            for x in range(9):
                string += str(self.grid[y, x]) + "|"
        string += "\n"
        string += "-" * 18

        return string

    def init_grid(self, grid):
        for y in range(9):
            for x in range(9):
                value = grid[y][x]
                self.grid[y, x] = value

    def is_filled(self):
        return self.isSolved

    def incr_last_apparition(self):
        self.last_apparition += 1

    def incr_nbr_apparition(self):
        self.nbr_apparition += 1

    def is_same_grid(self, points):
        thresh_dist = 0.03 * (self.w + self.h)
        points_grid = self.get_limits()
        for i in range(4):
            if np.linalg.norm(points_grid[i] - points[i]) > thresh_dist:
                return False

        self.last_apparition = 0
        self.set_limits(points)
        return True

    def is_same_grid_v2(self, points):
        thresh_dist = same_grid_dist_ratio * (self.w + self.h)
        is_same = []
        points_grid = self.get_limits()
        for i in range(4):
            is_same.append(np.linalg.norm(points_grid[i] - points[i]) < thresh_dist)

        if sum(is_same) < 3:
            return False

        if sum(is_same) == 3:
            false_value_ind = np.argmin(is_same)
            points[false_value_ind] = points_grid[false_value_ind]

        self.last_apparition = 0
        self.set_limits(points)
        return True
