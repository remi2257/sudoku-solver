import numpy as np
import copy


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
        if np.linalg.norm(self.TL - points[0]) > thresh_dist:
            return False
        if np.linalg.norm(self.TR - points[1]) > thresh_dist:
            return False
        if np.linalg.norm(self.BR - points[2]) > thresh_dist:
            return False
        if np.linalg.norm(self.BL - points[3]) > thresh_dist:
            return False

        self.last_apparition = 0
        self.set_limits(points)
        return True

