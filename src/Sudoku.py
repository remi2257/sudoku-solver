import numpy as np
import copy


class Sudoku:

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

    def apply_hypothesis_value(self, x, y, value):
        self.grid[y, x] = value
        self.possible_values_grid[y, x] = []

    def init_grid(self, grid):
        for y in range(9):
            for x in range(9):
                value = grid[y][x]
                self.grid[y, x] = value
                if value == 0:
                    self.possible_values_grid[y, x] = list(range(1, 10))
                else:
                    self.possible_values_grid[y, x] = []

    def is_filled(self):
        return 0 not in self.grid

    def get_possible_values(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:
                    self.count_possible_grid[y, x] = 0
                    continue
                possible_values = self.get_1_possible_values(x, y)
                self.possible_values_grid[y, x] = possible_values
                self.count_possible_grid[y, x] = len(possible_values)

    def get_1_possible_values(self, x, y):
        possible_values = self.possible_values_grid[y, x]
        self.check_line(y, possible_values)
        self.check_column(x, possible_values)
        self.check_square(x, y, possible_values)
        return possible_values

    def check_line(self, y, possible_values):
        line = self.grid[y, :]
        for value in reversed(possible_values):
            if value in line:
                possible_values.remove(value)

    def check_column(self, x, possible_values):
        column = self.grid[:, x]
        for value in reversed(possible_values):
            if value in column:
                possible_values.remove(value)

    def check_square(self, x, y, possible_values):
        x1 = 3 * (x // 3)
        y1 = 3 * (y // 3)
        x2, y2 = x1 + 3, y1 + 3
        square = self.grid[y1:y2, x1:x2]
        for value in reversed(possible_values):
            if value in square:
                possible_values.remove(value)

    def apply_unique_possibility(self):
        list_x_add = []
        list_y_add = []
        list_add_val = []
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] == 0 and self.count_possible_grid[y, x] == 1:
                    # print("Adding", self.possible_values_grid[y, x][0], "at", x, y)
                    value = self.possible_values_grid[y, x][0]
                    self.grid[y, x] = value
                    self.possible_values_grid[y, x] = []
                    list_x_add.append(x)
                    list_y_add.append(y)
                    list_add_val.append(value)

        if list_add_val == list(set(list_add_val)):
            return True
        return self.verify_new_result(zip(list_x_add, list_y_add))
        # return True

    def verify_new_result(self, my_zip):
        for x, y in my_zip:
            val = self.grid[y, x]
            self.grid[y, x] = 0
            line = self.grid[y, :]
            column = self.grid[:, x]
            x1 = 3 * (x // 3)
            y1 = 3 * (y // 3)
            x2, y2 = x1 + 3, y1 + 3
            square = self.grid[y1:y2, x1:x2]
            test = val in line or val in column or val in square
            self.grid[y, x] = val
            if test:
                return False

        return True

    '''
    def verify_new_result(self, my_zip):
        for x, y in my_zip:
            grid = copy.deepcopy(self.grid)
            grid[y, x] = 0
            line = grid[y, :]
            column = grid[:, x]
            x1 = 3 * (x // 3)
            y1 = 3 * (y // 3)
            x2, y2 = x1 + 3, y1 + 3
            square = grid[y1:y2, x1:x2]
            val = self.grid[y, x]
            if val in line or val in column or val in square:
                return False

        return True

    '''

    def should_make_hypothesis(self):
        return 1 not in self.count_possible_grid

    def best_hypothesis(self):
        count_less_options = 9
        best_x = 0
        best_y = 0
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:
                    continue
                if self.count_possible_grid[y, x] == 2:
                    return x, y, self.possible_values_grid[y, x]
                elif self.count_possible_grid[y, x] < count_less_options:
                    best_x, best_y = x, y
                    count_less_options = self.count_possible_grid[y, x]
                    if count_less_options == 0:
                        return None, None, []

        return best_x, best_y, self.possible_values_grid[best_y, best_x]

    def verify_result(self):
        # ok = True
        for y in range(9):
            for x in range(9):
                grid = copy.deepcopy(self.grid)
                grid[y, x] = 0
                line = grid[y, :]
                column = grid[:, x]
                x1 = 3 * (x // 3)
                y1 = 3 * (y // 3)
                x2, y2 = x1 + 3, y1 + 3
                square = grid[y1:y2, x1:x2]
                val = self.grid[y, x]
                if val in line or val in column or val in square:
                    # print(x, y)
                    # ok = False
                    return False

        return True
        # return ok


def verify_viable_grid(grid_tested):
    for y in range(9):
        for x in range(9):
            if grid_tested[y, x] == 0:
                continue
            grid = copy.deepcopy(grid_tested)
            grid[y, x] = 0
            line = grid[y, :]
            column = grid[:, x]
            x1 = 3 * (x // 3)
            y1 = 3 * (y // 3)
            x2, y2 = x1 + 3, y1 + 3
            square = grid[y1:y2, x1:x2]
            val = grid_tested[y, x]
            if val in line or val in column or val in square:
                # print("Unviable Grid")
                return False

    return True
