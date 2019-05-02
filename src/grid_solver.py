from src.Sudoku import *


def solve_grid(sudo):  # Return if grid is solved
    while not sudo.is_filled():
        sudo.get_possible_values()
        if sudo.should_make_hypothesis():
            # print('Hypothesis')
            x, y, possible_values_hyp = sudo.best_hypothesis()
            if not possible_values_hyp:  # At least one free spot can't have a solution
                return False, None
            for val in possible_values_hyp:
                new_sudo = Sudoku(sudo=sudo)
                new_sudo.apply_hypothesis_value(x, y, val)
                ret, solved_sudo = solve_grid(new_sudo)
                if ret:
                    return True, solved_sudo  # SOMETHING HAS BEEN SOLVED
                else:
                    del new_sudo
            return False, None  # None hypothesis lead to something
        else:
            ret = sudo.apply_unique_possibility()
            if ret is False:
                # print(sudo)
                # print("ARF")
                del sudo
                return False, None
        # print(sudo)
    # print("COMING HOME")
    return True, sudo


def main_solve_grids(grids):
    has_resolve_grid = False
    finished_grids = []
    for grid in grids:
        sudo = Sudoku(grid=grid)
        ret, finished_sudo = solve_grid(sudo)
        if ret:
            finished_grids.append(finished_sudo.grid)
            has_resolve_grid = True
        else:
            print("Failed during solving")
            # return None
    if has_resolve_grid:
        return finished_grids
    return None


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
        [0, 0, 0, 9, 0, 0, 7, 0, 0],
        [9, 0, 0, 3, 4, 0, 0, 0, 0],
        [2, 0, 0, 0, 1, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 7, 0],
        [0, 3, 0, 0, 2, 0, 0, 1, 0],
        [0, 5, 2, 0, 0, 9, 0, 0, 0],
        [0, 0, 8, 0, 6, 0, 0, 0, 5],
        [0, 0, 0, 0, 9, 1, 0, 0, 4],
        [0, 0, 4, 0, 0, 8, 0, 0, 0]
    ]

    target_grid = grid1
    init = time.time()
    f_sudo = main_solve_grids(target_grid)
    print("Took {:.5f} s".format(time.time() - init))
    print(Sudoku(grid=target_grid))
    print(f_sudo)
    print("Validated ?", Sudoku(grid=f_sudo).verify_result())
