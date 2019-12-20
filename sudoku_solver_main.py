import argparse
import os
import sys

from settings import logger
from tensorflow.keras.models import load_model

images_extension = [".jpg", ".jpeg", ".png", ".bmp", ".ash"]
video_extension = [".mp4", ".avi"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def help_profiles():
    return ('Profile type (Default: %(default)s)\n'
            '1 : Image/Video\n'
            '2 : Webcam /!\\ Too low quality won\'t lead to good result\n'
            )


def help_save():
    return ("Save output result in a repo folder"
            "(images_save/ for images || videos_result/ for a Video)\n"
            "(Default: %(default)s)\n"
            '0 : No save\n'
            '1 : [IMAGE] Save result / [VIDEO] Save full result\n'
            '2 : [VIDEO] Save only frames with detected grid in a .gif\n'
            '3 : [VIDEO] Save everything in a .gif + multiple .jpg\n'
            )


def description():
    warning = 15 * '-' + ' /!\\Quality Work/!\\ ' + 16 * '-'

    welcome_messsage = 'Welcome on Sudoku Solver'
    new_welcome = ' ' * max(int((len(warning) - len(welcome_messsage)) / 2), 0) + welcome_messsage
    little_description = "Sudoku Solver is a cool tool which has 1 only goal :" \
                         "\n-> Solve a Sudoku"

    return warning + "\n" + new_welcome + "\n" + warning + "\n" + little_description + "\n" + warning


def parser_generation():
    parser = argparse.ArgumentParser(description=description(),
                                     epilog=40 * '-' + '\n Rémi LUX\n' + 40 * '-',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-i", "--i_path",
                        help="Path of the input image/video",
                        )
    parser.add_argument("-p", "--profile", help=help_profiles(),
                        type=int, choices=[1, 2, 3], default=1)

    parser.add_argument("-mp", "--model_path",
                        help="Path of the CNN model to identify digits", type=str,
                        default='model/my_model.h5')

    parser.add_argument("-s", "--save",
                        help=help_save(),
                        type=int, choices=[0, 1, 2, 3], default=1)

    parser.add_argument("-d", "--display",
                        help="display output detail",
                        action="store_true")

    parser.add_argument("-v", "--verbose", default='critical', type=str.lower,
                        choices=['critical', 'error','warning', 'info', 'debug'],
                        help="Set verbose Level (Default: %(default)s)"
                        )

    args = parser.parse_args()

    return args


def setting_recup():  # Function to read parameter settings
    args = parser_generation()
    logger.setLevel(args.verbose.upper())

    if args.i_path is None and not args.profile == 2:
        logger.warning("\nCannot analyse an image neither a video without a path\nSwitching to Webcam...")
        args.profile = 2

    try:
        model = load_model(args.model_path)
    except OSError:
        logger.critical("\nCannot localize model\nLeaving...")
        sys.exit(3)
    return args, model


def main_process():
    args, model = setting_recup()
    if args.profile == 1:  # Image or Video
        if args.i_path.endswith(tuple(images_extension)):
            from src.main_img import main_process_img
            main_process_img(args.i_path, model, args.save)
        elif args.i_path.endswith(tuple(video_extension)):
            from src.main_video import main_grid_detector_video
            main_grid_detector_video(model, video_path=args.i_path, save=args.save, display=args.display)
        else:
            logger.critical("\nCannot identify File type\nLeaving...")
            sys.exit(3)
    else:  # Webcam
        from src.main_video import main_grid_detector_video
        main_grid_detector_video(model, save=args.save, display=args.display)  #


if __name__ == '__main__':
    main_process()
