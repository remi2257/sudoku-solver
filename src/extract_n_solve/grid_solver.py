from src.solving_objects.Sudoku import *


class GridSolver:
    def __init__(self):
        pass

    def solve_grids(self, grids, hint_mode=False):
        return [self.solve_grid(grid, hint_mode) for grid in grids]

    def solve_grid(self, grid, hint_mode=False):
        if grid is None:
            return None
        sudo = Sudoku(grid=grid)
        ret, finished_sudo = self.process_solving_grid(sudo)
        if ret:
            if not hint_mode:
                return finished_sudo.grid
            else:
                return sudo.give_an_hint()
        else:
            return None

    def process_solving_grid(self, sudo):  # Return if grid is solved
        while not sudo.is_filled():
            # sudo.get_possible_values()
            if sudo.should_make_hypothesis():
                x, y, possible_values_hyp = sudo.best_hypothesis()
                if not possible_values_hyp:  # At least one free spot can't have a solution
                    return False, None
                for val in possible_values_hyp:
                    new_sudo = Sudoku(sudo=sudo)
                    new_sudo.apply_hypothesis_value(x, y, val)
                    ret, solved_sudo = self.process_solving_grid(new_sudo)
                    if ret:
                        return True, solved_sudo  # SOMETHING HAS BEEN SOLVED
                    else:
                        del new_sudo
                return False, None  # None hypothesis lead to something
            else:
                ret = sudo.apply_unique_possibilities()
                if ret is False:
                    # print(sudo)
                    # print("ARF")
                    del sudo
                    return False, None
        # print("COMING HOME")
        return True, sudo


if __name__ == '__main__':
    import time

    grid1 = [
        [8, 7, 6, 9, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 6, 0, 0, 0],
        [0, 4, 0, 3, 0, 5, 8, 0, 0],
        [4, 0, 0, 0, 0, 0, 2, 1, 0],
        [0, 9, 0, 5, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 4, 0, 3, 0, 6],
        [0, 2, 9, 0, 0, 0, 0, 0, 8],
        [0, 0, 4, 6, 9, 0, 1, 7, 3],
        [0, 0, 0, 0, 0, 1, 0, 0, 4]
    ]
    grid2 = [
        [0, 0, 4, 0, 0, 6, 0, 8, 0],
        [0, 7, 0, 0, 5, 0, 0, 3, 4],
        [0, 0, 2, 0, 0, 4, 7, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0],
        [6, 0, 0, 3, 0, 7, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 7, 8, 0, 0, 3, 0, 0],
        [1, 3, 0, 0, 4, 0, 0, 6, 0],
        [0, 8, 0, 6, 0, 0, 1, 0, 0]
    ]
    grid3 = [
        [0, 0, 0, 0, 0, 7, 0, 0, 0],
        [0, 6, 4, 9, 2, 8, 5, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 9],
        [5, 0, 0, 0, 8, 0, 9, 2, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 9, 6, 0, 4, 0, 0, 0, 5],
        [9, 5, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 3, 6, 0, 0, 2, 5, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0]
    ]
    grid4 = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ]

    target_grid = grid1
    init = time.time()
    solver = GridSolver()
    f_sudo = solver.solve_grid(target_grid)
    print(Sudoku(grid=target_grid))
    if f_sudo is None:
        print("echec")
    else:
        sudo_res = Sudoku(grid=f_sudo)
        print(sudo_res)
        print("Validated ?", sudo_res.verify_result())

    print("Took {:.1f} ms".format(1000 * (time.time() - init)))
