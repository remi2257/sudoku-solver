import os


def rename():
    file_list = [path for path in os.listdir("./") if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{:09d}.jpg".format(i))
    file_list = [path for path in os.listdir("./") if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{:03d}.jpg".format(i))


if __name__ == '__main__':
    rename()
