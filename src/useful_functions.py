from time import time

import cv2

from settings import *

keys_leave = [27, ord('q')]


def timer_decorator(func):
    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print("{:05.2f}ms elapsed in {}".format(1000 * (after - before), func.__name__))
        return rv

    return f


# @timer_decorator
def my_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):  # INTER_AREA
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        if height is None:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            if w < h:
                r = width / float(w)
                dim = (width, int(h * r))
            else:
                r = height / float(h)
                dim = (int(w * r), height)
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


width_target = 640


def get_video_save_size(h, w):
    r = width_target / float(w)
    dim = (width_target, int(h * r))

    return dim


def find_available_camera_port():
    """
    Run on all over possible ports to find cameras' port
    :return: list of viable ports
    """
    import time

    list_viable = []
    for source in range(8):
        cap = cv2.VideoCapture(source)
        # print(cap.isOpened())
        if cap is None or not cap.isOpened():
            # print('Warning: unable to open video source: ', source)
            pass
        else:
            list_viable.append(source)
    time.sleep(0.1)
    return list_viable


def show_and_check_close(window_name, image):
    """
    Display an image and check whether the user want to close
    :param window_name: window's name
    :param image: Image
    :return: boolean indicating if the user wanted to leave
    """
    cv2.imshow(window_name, image)
    return cv2.waitKey(1) in keys_leave


def show_and_wait_close(window_name, image):
    """
    Display an image and wait that the user close it
    :param window_name: window's name
    :param image: Image
    :return: None
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def add_annotation_to_image(img, text, write_on_top=True):
    """
    Add Annotation to an image
    :param img: Image
    :param text: text string
    :param write_on_top: if you write the text on top
    :return: img with text written on it
    """
    font_scale_used = font_scale_small
    thickness_used = thickness_small
    h_im, w_im = img.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, font_base, fontScale=font_scale_used,
                                                          thickness=thickness_used)
    text_true_height = text_height + baseline
    if write_on_top:
        cv2.rectangle(img, (0, 0),
                      (round(text_width * 1.1), round(text_true_height * 1.35)),
                      WHITE, cv2.FILLED)
        cv2.putText(img, text,
                    (round(text_width * 0.05), round(text_true_height * 1.2 - baseline)),
                    font_base, font_scale_used, ORANGE, thickness_used)
    else:
        cv2.rectangle(img,
                      (round(text_width * 1.1), h_im - round(text_true_height * 1.35)),
                      (0, h_im),
                      WHITE, cv2.FILLED)
        cv2.putText(img, text,
                    (round(text_width * 0.05), h_im - baseline),
                    font_base, font_scale_used, ORANGE, thickness_used)
    return img


def create_gif(video_path, o_folder, fps=5, width=360):
    gif_file_name = o_folder + "video_solve_0.gif"
    ind_save = 0
    while os.path.isfile(gif_file_name):
        ind_save += 1
        gif_file_name = o_folder + "video_solve_{}.gif".format(ind_save)

    cmd = ["ffmpeg",
           "-i", video_path,
           "-vf scale={}:-1".format(width),
           "-r", str(fps),
           "-f image2pipe",
           "-loglevel fatal",
           "-vcodec ppm - |",
           "convert",
           "-delay", str(20),
           "-loop 0",
           "-", gif_file_name
           ]
    cmd = " ".join(cmd)
    os.system(cmd)


if __name__ == '__main__':
    video_path_ = "/home/remi/PycharmProjects/sudoku-solver/videos_result/out_process_1.mp4"
    o_folder_ = "/home/remi/PycharmProjects/sudoku-solver/videos_result/"
    create_gif(video_path_, o_folder_, width=480)
    # im = cv2.imread("images_test/sudoku2.jpg")
    # my_resize(im, height=900)
    # my_resize(im, height=900, inter=cv2.INTER_AREA)
