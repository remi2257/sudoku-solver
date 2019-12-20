from tensorflow.keras.models import load_model

from src.extract_n_solve.extract_digits import process_extract_digits_single
from src.extract_n_solve.grid_detector_img import main_grid_detector_img
from src.extract_n_solve.grid_solver import main_solve_grid
from src.extract_n_solve.new_img_generator import *
from src.solving_objects.SudokuVideo import *
from src.useful_functions import *
from src.video_objects.WebcamVideoStream import *


def update_sudoku_lists(list_possible_grid, using_webcam=False):
    for i in reversed(range(len(list_possible_grid))):
        list_possible_grid[i].incr_last_apparition()
        lim_time = (1 + int(using_webcam)) * (
            lim_apparition_solved if list_possible_grid[i].isSolved else lim_apparition_not_solved)
        if list_possible_grid[i].last_apparition > lim_time:
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


def extract_digits_and_solve(im_grids_final, model, old_solution_list, list_possible_grid, points_grids):
    # print(len(list_possible_grid))
    grids_matrix = []
    for im_grid_final, old_solution in zip(im_grids_final, old_solution_list):
        if old_solution == -1:
            grids_matrix.append(process_extract_digits_single(im_grid_final, model,
                                                              save_image_digits=False))
        else:
            grids_matrix.append(old_solution.grid)
    if all(elem is None for elem in grids_matrix):
        return None, None
    grids_solved = []
    for grid_matrix, old_solution, points_grid in zip(grids_matrix, old_solution_list, points_grids):
        if grid_matrix is None:
            grids_solved.append(None)
            continue
        if old_solution != -1:
            grids_solved.append(old_solution.grid_solved)
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
                # pass
    return grids_matrix, grids_solved


def show_im_final(im_final, init_time):
    add_annotation_to_image(im_final, "{:.2f} FPS".format(1 / (time.time() - init_time)), write_on_top=True)
    cv2.imshow("im_final", im_final)


# def create_windows(display):
#     w_window, h_window = 800, 500
#     wind_names = []
#     if display:
#         # wind_names = ['frame_resize', 'img_lines', 'img_contour']  # ,'prepro_im'
#         wind_names = ['res']  # ,'prepro_im'
#     wind_names += ['im_final']
#
#     for i, wind_name in enumerate(wind_names):
#         cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(wind_name, w_window, h_window)
#         new_x = (i // 2) * w_window * 1.1
#         new_y = (i % 2) * h_window * 1.2
#         cv2.moveWindow(wind_name, int(new_x), int(new_y))

def create_windows(display):
    if display:
        cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('res', 800, 800)
        cv2.moveWindow('res', 900, 0)

    w_window, h_window = 800, 500
    cv2.namedWindow('im_final', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('im_final', w_window, h_window)


lim_frames_without_grid = 2
save_folder = "videos_result/"


def main_grid_detector_video(model, video_path=None, save=0, display=True):
    # Initialisation
    list_possible_grid, ims_filled_grid = [], None
    grids_solved = None
    points_grids_saved = list()
    list_matrix_saved = list()
    video_out_path = save_folder + 'out_process_0.mp4'
    ind_save = 0
    while os.path.isfile(video_out_path):
        ind_save += 1
        video_out_path = save_folder + 'out_process_{}.mp4'.format(ind_save)

    frames_without_grid = 0
    create_windows(display=display)
    using_webcam = video_path is None

    # Starting streaming
    if using_webcam:  # Use Webcam
        cap = WebcamVideoStream().start()
    else:
        cap = cv2.VideoCapture(video_path)
    output_video = None

    w_vid_target, h_vid_target = get_video_save_size(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                                     cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        output_video = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                       # (w_vid_target, h_vid_target))
                                       (output_width, output_height))
        # (1920, 1080))

    while "User do not stop":
        # -- Init the iteration
        if cv2.waitKey(1) in keys_leave:
            break

        init_time = time.time()

        if using_webcam:
            img = cap.read()
        else:
            ret, img = cap.read()
            if not ret:
                break

        img_final = my_resize(img, height=output_height)
        ratio = float(img_final.shape[0]) / img.shape[0]
        # Update Suodku Lists (for keeping already detected grids & deleting old ones)
        update_sudoku_lists(list_possible_grid, using_webcam=using_webcam)

        read_time = time.time()
        logger.info("{}\nread&update_time \t{:05.2f}ms".format("-" * 20, 1000 * (read_time - init_time)))

        # --- LOOKING FOR GRIDS
        # Try to detect grids !
        im_grids_final, points_grids, list_transform_matrix = main_grid_detector_img(img, display=display,
                                                                                     resized=False,
                                                                                     using_webcam=using_webcam)
        grid_detection_time = time.time()
        logger.info("grid_detect_time \t{:05.2f}ms".format(1000 * (grid_detection_time - read_time)))

        if im_grids_final is None:  # No grid has been found
            frames_without_grid += 1
            if frames_without_grid < lim_frames_without_grid and grids_solved is not None:
                img_final = recreate_img_filled(img_final, ims_filled_grid,
                                                points_grids_saved, list_matrix_saved,
                                                ratio=ratio)
            show_im_final(img_final, init_time)

            if save == 1:
                output_video.write(img_final)
            if save == 3:
                cv2.imwrite("tmp/{:03d}.jpg".format(ind_save), img)
                ind_save += 1
            continue

        logger.debug("{} grid(s) detected".format(len(im_grids_final)))

        # -- EXTRACTING GRIDS
        points_grids_saved = points_grids.copy()
        list_matrix_saved = list_transform_matrix.copy()

        # Checking if grids has been already solved with position
        old_solution_list = look_for_already_solved_grid(im_grids_final, list_possible_grid, points_grids)
        # Extracting & solving
        grids_matrix, grids_solved = extract_digits_and_solve(im_grids_final, model, old_solution_list,
                                                              list_possible_grid,
                                                              points_grids)
        solving_time = time.time()
        logger.info("solving_time \t\t{:05.2f}ms".format(1000 * (solving_time - grid_detection_time)))
        if grids_solved is None:
            show_im_final(img_final, init_time)
            if save == 1:
                output_video.write(img_final)
            if save == 3:
                cv2.imwrite("tmp/{:03d}.jpg".format(ind_save), img)
                ind_save += 1
            continue
        frames_without_grid = 0
        ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
        write_time = time.time()
        logger.info("write_time \t\t{:05.2f}ms".format(1000 * (write_time - solving_time)))

        img_final = recreate_img_filled(img_final, ims_filled_grid,
                                        points_grids, list_transform_matrix, ratio=ratio)
        recreation_time = time.time()
        logger.info("recreation_time \t{:05.2f}ms".format(1000 * (recreation_time - write_time)))

        show_im_final(img_final, init_time)
        show_time = time.time()
        logger.info("show_time \t\t{:05.2f}ms".format(1000 * (show_time - recreation_time)))

        if save > 0:
            # output_video.write(my_resize(img_final, w_vid_target))
            output_video.write(img_final)
            # if save == 3:
            #     cv2.imwrite("tmp/{:03d}.jpg".format(ind_save), frame)
            #     ind_save += 1
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()
    if save:
        try:
            output_video.release()
            print("Saving GIF ....")
            create_gif(video_out_path, save_folder)
        except AttributeError:
            logger.warning("Cannot release output_video")


if __name__ == '__main__':
    my_model = load_model('model/my_model.h5')
    video_p = "/media/hdd_linux/Video/sudoku1.mp4"
    main_grid_detector_video(my_model, video_path=video_p,
                             save=1, display=True)
    # main_grid_detector_video(my_model)
