from src.grid_solver import main_solve_grids
from src.new_img_generator import *
from src.extract_digits import process_extract_digits
from src.grid_detector_img import resize, get_p_hough_transform, preprocess_im, undistorted_grids
from keras.models import load_model


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
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not resized:
        frame_resize = resize(frame, height=900, width=900)
    else:
        frame_resize = frame
    ratio = frame.shape[0] / frame_resize.shape[0]
    prepro_im = preprocess_im(frame_resize)
    if display:
        extreme_points, img_lines, img_contour = get_p_hough_transform(frame_resize.copy(), prepro_im, display)
        cv2.imshow('frame_resize', frame_resize)
        # cv2.imshow('prepro_im', prepro_im)
        cv2.imshow('img_lines', img_lines)
        cv2.imshow('img_contour', img_contour)
    else:
        extreme_points = get_p_hough_transform(frame_resize.copy(), prepro_im)

    if extreme_points is None:
        return None, None

    grids_extracted_final, points_grids = undistorted_grids(frame_resize, extreme_points, ratio)
    return grids_extracted_final, points_grids


def main_grid_detector_video(display=True, video_path=None):
    create_windows(display)
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    while "user does not exit":
        if cv2.waitKey(1) & 0xFF == 27:  # ord('q')
            break
        # Capture frame-by-frame
        _, frame = cap.read()
        # cv2.imshow("frame", frame)

        im_grids_final, points_grids = grid_detector(frame, display=display)
        
        if im_grids_final is None:
            continue
        # for (i, im_grid) in enumerate(im_grids_final):
        #     cv2.imshow('grid_{}'.format(i), im_grid)
        grids_matrix = process_extract_digits(im_grids_final, model)
        if all(elem is None for elem in grids_matrix):
            continue
        grids_solved = main_solve_grids(grids_matrix)
        if grids_solved is None:
            continue
        ims_filled_grid = write_solved_grids(im_grids_final, grids_matrix, grids_solved)
        im_final = recreate_img_filled(frame, ims_filled_grid, points_grids)
        cv2.imshow("im_final", im_final)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = load_model('model/my_model.h5')
    video_p = "/media/hdd_linux/Video/sudoku1.mp4"
    main_grid_detector_video(model, video_path=video_p)
    # main_grid_detector_video(model)
