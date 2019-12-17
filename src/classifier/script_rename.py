import os
import shutil

def rename(target_path, offset=0):
    file_list = [target_path + path for path in os.listdir(target_path) if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{}{:08d}.jpg".format(target_path, i))

    file_list = [target_path + path for path in os.listdir(target_path) if path.endswith(".jpg")]
    i = 0
    for file in file_list:
        i += 1
        os.rename(file, "{}{:03d}.jpg".format(target_path, i + offset))


def move_folder_to_another(initial, final):
    file_list = [initial + path for path in os.listdir(initial) if path.endswith(".jpg")]
    output_file_list = [final + path for path in os.listdir(initial) if path.endswith(".jpg")]

    for input_path, output in zip(file_list, output_file_list):
        shutil.copy(input_path, output)
        # print(input_path,output)
    rename(final)


if __name__ == '__main__':
    for i in range(1,10):
        path_rename = "/media/hdd_linux/DataSet/Mine/{}/".format(i)
        rename(path_rename)
    path_rename = "/media/hdd_linux/DataSet/Mine/N/"
    rename(path_rename)
    # move_folder_to_another(
    #     initial="/media/hdd_linux/DataSet/Mine/save/N/",
    #     final="/media/hdd_linux/DataSet/Mine/N/"
    # )