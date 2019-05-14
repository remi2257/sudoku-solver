from src.grid_solver import main_solve_grid
from src.new_img_generator import *
from src.SudokuVideo import *
from src.extract_digits import process_extract_digits_single
from src.grid_detector_img import get_p_hough_transform, preprocess_im, undistorted_grids
from keras.models import load_model
from src.fonctions import *
import time

lim_apparition_not_solved = 12
lim_apparition_solved = 60


def update_sudo_lists(list_possible_grid):
    for i in reversed(range(len(list_possible_grid))):
        list_possible_grid[i].incr_last_apparition()
        if list_possible_grid[i].isSolved:
            if list_possible_grid[i].last_apparition > lim_apparition_solved:
                del list_possible_grid[i]
        else:
            if list_possible_grid[i].last_apparition > lim_apparition_not_solved:
                del list_possible_grid[i]


def look_for_already_solved_grid(im_grids_final, list_possible_grid, points_grids):
    already_solved = [-1] * len(im_grids_final)
    good_grids = [grid for grid in list_possible_grid if grid.isSolved]

    if good_grids:
        for i, points_grid in enumerate(points_grids):
            for good_grid in good_grids:
                if good_grid.last_apparition < lim_apparition_not_solved and good_grid.is_same_grid(points_grid):
                    already_solved[i] = good_grid
                    continue
    return already_solved


thresh_apparition_conf = 1


def extract_and_solve(im_grids_final, model, already_solved_list, list_possible_grid, points_grids):
    # print(len(list_possible_grid))
    grids_matrix = []
    for im_grid_final, already_solved in zip(im_grids_final, already_solved_list):
        if already_solved == -1:
            grids_matrix.append(process_extract_digits_single(im_grid_final, model))
        else:
            grids_matrix.append(already_solved.grid)
    if all(elem is None for elem in grids_matrix):
        return None, None
    grids_solved = []
    for grid_matrix, already_solved, points_grid in zip(grids_matrix, already_solved_list, points_grids):
        if grid_matrix is None:
            grids_solved.append(None)
            continue
        if already_solved != -1:
            grids_solved.append(already_solved.grid_solved)
        else:
            has_been_found = False
            for sudoku in list_possible_grid:
                # print("grid_matrix",grid_matrix)
                # print("grids_raw",sudoku.grid_raw)
                if np.array_equal(grid_matrix, sudoku.grid_raw):
                    has_been_found = True
                    sudoku.incr_nbr_apparition()
                    sudoku.set_limits(points_grid)
                    if sudoku.nbr_apparition > thresh_apparition_conf:
                        sudoku.isConfident = True
                        if not sudoku.isSolved:
                            sudoku.grid_solved = main_solve_grid(grid_matrix)
                        if sudoku.grid_solved is not None:
                            sudoku.isSolved = True
                        else:
                            sudoku.last_apparition = 1000  # Impossible grid .. Will be delete next time
                        grids_solved.append(sudoku.grid_solved)
                    break
            if not has_been_found:
                list_possible_grid.append(SudokuVideo(grid_matrix))
                grids_solved.append(None)

    return grids_matrix, grids_solved


def write_FPS(im, elapsed_time):
    cv2.putText(im, "{:.2f} FPS".format(1 / elapsed_time),
                (50, 80), font, 3, (0, 255, 0), thickness)


def show_im_final(im_final, init_time):
    write_FPS(im_final, time.time() - init_time)
    cv2.imshow("im_final", im_final)


def create_windows(display):
    w_window, h_window = 800, 500
    wind_names = []
    if display:
        wind_names = ['frame_resize', 'img_lines', 'img_contour']  # ,'prepro_im'
    wind_names += ['im_final']

    for i, wind_name in enumerate(wind_names):
        cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wind_name, w_window, h_window)
        new_x = (i // 2) * w_window * 1.1
        new_y = (i % 2) * h_window * 1.2
        cv2.moveWindow(wind_name, int(new_x), int(new_y))


def grid_detector(frame, display=False, resized=False):
    # t0 = time.time()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not resized:
        frame_resize = resize(frame, height=900, width=900)
    else:
        frame_resize = frame
    ratio = frame.shape[0] / frame_resize.shape[0]
    prepro_im = preprocess_im(frame_resize)
    # t1 = time.time()

    if display:
        extreme_points, img_lines, img_contour = get_p_hough_transform(frame_resize.copy(), prepro_im, display)
        cv2.imshow('frame_resize', frame_resize)
        # cv2.imshow('prepro_im', prepro_im)
        cv2.imshow('img_lines', img_lines)
        cv2.imshow('img_contour', img_contour)
    else:
        extreme_points = get_p_hough_transform(frame_resize.copy(), prepro_im)
    # t2 = time.time()

    if extreme_points is None:
        return None, None

    grids_extracted_final, points_grids = undistorted_grids(frame_resize, extreme_points, ratio)
    # t3 = time.time()

    # total_time = t3 - t0
    # prepro_time = t1 - t0
    # print("prepro_time \t\t{:.1f}% - {:.3f}s".format(100 * prepro_time / total_time, prepro_time))
    # hough_time = t2 - t1
    # print("Hough Transfrom \t{:.1f}% - {:.3f}s".format(100 * hough_time / total_time, hough_time))
    # undistort_time = t3 - t2
    # print("undistort_time \t\t{:.1f}% - {:.3f}s".format(100 * undistort_time / total_time, undistort_time))
    # print("EVERYTHING DONE \t{:.2f}s".format(total_time))

    return grids_extracted_final, points_grids


lim_frames_without_grid = 2
save_folder = "videos_result/"


def main_grid_detector_video(model, video_path=None, save=0, display=True):
    list_possible_grid, ims_filled_grid = [], None
    grids_solved = None
    video_out_path = save_folder + 'out_process_0.mp4'
    ind_save = 0
    while os.path.isfile(video_out_path):
        ind_save += 1
        video_out_path = save_folder + 'out_process_{}.mp4'.format(ind_save)

    frames_without_grid = 0
    create_windows(display)
    if video_path is None:  # Use Webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    output_video = None

    w_vid_target, h_vid_target = get_video_save_size(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                                     cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        output_video = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                       (w_vid_target, h_vid_target))
        # (1920, 1080))

    while "User do not stop":
        if cv2.waitKey(1) & 0xFF == 27:  # ord('q')
            break

        init_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        update_sudo_lists(list_possible_grid)
        im_grids_final, points_grids = grid_detector(frame, display=display)

        # if im_grids_final is None:
        #     frames_without_grid += 1
        #     if frames_without_grid < lim_frames_without_grid and grids_solved is not None:
        #         im_final = recreate_img_filled(frame, ims_filled_grid, points_grids)
        #         show_im_final(im_final, init_time)
        #     else:
        #         show_im_final(frame, init_time)
        #     continue
        if im_grids_final is None:
            if save == 1:
                output_video.write(resize(frame, w_vid_target))
                # output_video.write(frame)
            frames_without_grid += 1
            show_im_final(frame, init_time)
            continue

        already_solved = look_for_already_solved_grid(im_grids_final, list_possible_grid, points_grids)
        # for (i, im_grid) in enumerate(im_grids_final):
        #     cv2.imshow('grid_{}'.format(i), im_grid)
        grids_matrix, grids_solved = extract_and_solve(im_grids_final, model, already_solved, list_possible_grid,
                                                       points_grids)

        if grids_solved is None:
            if save == 1:
                output_video.write(resize(frame, w_vid_target))
                # output_video.write(frame)
            show_im_final(frame, init_time)
            continue
        frames_without_grid = 0
        ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
        im_final = recreate_img_filled(frame, ims_filled_grid, points_grids)
        if save > 0:
            output_video.write(resize(im_final, w_vid_target))
            cv2.imwrite("tmp/{:03d}.jpg".format(ind_save), frame)
            ind_save += 1
            # output_video.write(frame)
        show_im_final(im_final, init_time)
    # When everything done, release the capture
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()

    if save:
        create_gif(video_out_path, save_folder)
        # shutil.rmtree(im_temp_folder, ignore_errors=True)


if __name__ == '__main__':
    my_model = load_model('model/my_model.h5')
    video_p = "/media/hdd_linux/Video/sudoku1.mp4"
    main_grid_detector_video(my_model, video_path=video_p, save=1, display=True)
    # main_grid_detector_video(model)
