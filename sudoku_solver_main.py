import argparse
import sys


def help_profiles():
    return ('Profile type (Default: %(default)s)\n'
            '1 : Image\n'
            '2 : Video\n'

            )


def description():
    warning = 15 * '-' + ' /!\\Quality Work/!\\ ' + 16 * '-'

    welcome_messsage = 'Welcome on Sudoku Solver'
    new_welcome = ' ' * max(int((len(warning) - len(welcome_messsage)) / 2), 0) + welcome_messsage
    little_description = "Sudoku Solver is a cool tool which has 1 only goal :" \
                         "\n-> Solve a Sudoku (lol)"

    return warning + "\n" + new_welcome + "\n" + warning + "\n" + little_description + "\n" + warning


def parser_generation():
    parser = argparse.ArgumentParser(description=description(),
                                     epilog=40 * '-' + '\n Rémi LUX\n' + 40 * '-',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-p", "--profile", help=help_profiles(),
                        type=int, choices=[1, 2], default=1)
    parser.add_argument("-i", "--ifile", help="Path of the input image / Necessary with Profile 1",
                        )
    parser.add_argument("-s", "--save", help="Save output image",
                        action="store_true")

    args = parser.parse_args()

    return args


def setting_recup():  # Function to read parameter settings
    args = parser_generation()

    if args.profile == 1 and args.ifile is None:
        print("Cannot analyse an image without a path")
        sys.exit(3)

    return args


def main_process():
    args = setting_recup()

    from src.main_img import main_process_img
    main_process_img(args.ifile, args.save)


if __name__ == '__main__':
    main_process()
