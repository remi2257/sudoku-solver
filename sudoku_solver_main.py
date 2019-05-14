import argparse
import sys
import tensorflow as tf
import os

images_extension = [".jpg", ".jpeg", ".png", ".bmp", ".ash"]
video_extension = [".mp4", ".avi"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def help_profiles():
    return ('Profile type (Default: %(default)s)\n'
            '1 : Image/Video\n'
            '2 : Webcam /!\\ Too low quality won\'t lead to good result\n'
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

    parser.add_argument("-s", "--save", help="Save output result in a repo folder"
                                             "(images_save/ for images videos_result/ for a Video)",
                        action="store_true")

    args = parser.parse_args()

    return args


def setting_recup():  # Function to read parameter settings
    args = parser_generation()

    if args.i_path is None and not args.profile == 2:
        print("\nCannot analyse an image|video without a path\nLeaving...")
        sys.exit(3)

    try:
        from keras.models import load_model
        model = load_model(args.model_path)
    except OSError:
        print("\nCannot localize model\nLeaving...")
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
            main_grid_detector_video(model, video_path=args.i_path, save=int(args.save))
        else:
            print("\nCannot identify File type\nLeaving...")
            sys.exit(3)
    else:  # Webcam
        from src.main_video import main_grid_detector_video
        main_grid_detector_video(model, save=int(args.save))


if __name__ == '__main__':
    main_process()
