import os


def rename(target_path):
    file_list = [target_path + path for path in os.listdir(target_path) if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{}{:08d}.jpg".format(target_path, i))

    file_list = [target_path + path for path in os.listdir(target_path) if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{}{:03d}.jpg".format(target_path, i))


if __name__ == '__main__':
    path_rename = "dataset_test/"
    rename(path_rename)
