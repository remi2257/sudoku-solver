import numpy as np
import copy


class SudokuImage:

    def __init__(self, sudo=None, grid=None):
        if sudo is None:
            self.grid = np.zeros((9, 9), dtype=int)
            self.possible_values_grid = np.empty((9, 9), dtype=list)
            self.init_grid(grid)
            self.count_possible_grid = np.zeros((9, 9), dtype=int)
        else:
            self.grid = copy.deepcopy(sudo.grid)
            self.possible_values_grid = copy.deepcopy(sudo.possible_values_grid)
            self.count_possible_grid = copy.deepcopy(sudo.count_possible_grid)

    def __str__(self):
        string = "-" * 18
        for y in range(9):
            string += "\n|"
            for x in range(9):
                string += str(self.grid[y, x]) + "|"
        string += "\n"
        string += "-" * 18

        return string