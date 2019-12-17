import os
import random
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset_path = "/media/hdd_linux/DataSet/Mine/"
img_size = 28 * 28


def create_csv(data_path):
    folders_list = [data_path + path for path in os.listdir(data_path) if os.path.isdir(data_path + path)]

    folders_list = [folder for folder in folders_list if not folder.endswith(tuple(["train", "test","temp"]))]
    folders_list = sorted(folders_list, key=lambda x: os.path.basename(x))
    # df = pd.DataFrame({"label": [],
    #                    "images_test": []})
    columns = ['label'] + ["pixel" + str(i) for i in range(img_size)]
    df = pd.DataFrame(columns=columns)

    for folder in tqdm(folders_list, total=len(folders_list), desc="Big Loop"):
        name = os.path.basename(folder)[0]
        label = 10 if name == "N" else int(name)
        folder += "/"
        file_list = [folder + im_path for im_path in os.listdir(folder) if os.path.isfile(folder + im_path)]
        # file_list = file_list[:200]
        nbr_img = len(file_list)
        df2 = pd.DataFrame(columns=columns, index=list(range(0, nbr_img)))
        for i in tqdm(range(nbr_img), total=len(file_list), desc="Small Loop"):
            img = cv2.imread(file_list[i], cv2.IMREAD_GRAYSCALE)
            im_resize = cv2.resize(img, (28, 28))
            im_resize = np.reshape(im_resize, -1)
            df2.loc[i][0] = label
            for k in range(img_size):
                df2.loc[i][k + 1] = im_resize[k]

        df = pd.concat([df, df2])

    df.to_csv(dataset_path + 'minst_train_test.csv')


def create_csv_seperated(data_path):
    folders_list = [data_path + path for path in os.listdir(data_path) if os.path.isdir(data_path + path)]

    folders_train = [folder for folder in folders_list if folder.endswith("train")]
    folders_train = sorted(folders_train, key=lambda x: int(os.path.basename(x)[0]))
    folders_test = [folder for folder in folders_list if folder.endswith("test")]
    folders_test = sorted(folders_test, key=lambda x: int(os.path.basename(x)[0]))
    # df = pd.DataFrame({"label": [],
    #                    "images_test": []})
    columns = ['label'] + ["pixel" + str(i) for i in range(img_size)]
    df = pd.DataFrame(columns=columns)

    for folder in tqdm(folders_train, total=len(folders_train), desc="Big Loop Train"):
        label = int(os.path.basename(folder)[0])
        folder += "/"
        file_list = [folder + im_path for im_path in os.listdir(folder) if os.path.isfile(folder + im_path)]
        # file_list = file_list[:200]
        nbr_img = len(file_list)
        df2 = pd.DataFrame(columns=columns, index=list(range(0, nbr_img)))
        for i in tqdm(range(nbr_img), total=len(file_list), desc="Small Loop"):
            img = cv2.imread(file_list[i], cv2.IMREAD_GRAYSCALE)
            im_resize = cv2.resize(img, (28, 28))
            im_resize = np.reshape(im_resize, -1)
            df2.loc[i][0] = label
            for k in range(img_size):
                df2.loc[i][k + 1] = im_resize[k]

        df = pd.concat([df, df2])

    df.to_csv(dataset_path + 'minst_train.csv')

    # df = pd.DataFrame({"label": [],
    #                    "images_test": []})
    columns = ['label'] + ["pixel" + str(i) for i in range(img_size)]
    df = pd.DataFrame(columns=columns)

    for folder in tqdm(folders_test, total=len(folders_test), desc="Big Loop Test"):
        label = int(os.path.basename(folder)[0])
        folder += "/"
        file_list = [folder + im_path for im_path in os.listdir(folder) if os.path.isfile(folder + im_path)]
        # file_list = file_list[:30]
        nbr_img = len(file_list)
        df2 = pd.DataFrame(columns=columns, index=list(range(0, nbr_img)))
        for i in tqdm(range(nbr_img), total=len(file_list), desc="Small Loop"):
            img = cv2.imread(file_list[i], cv2.IMREAD_GRAYSCALE)
            im_resize = cv2.resize(img, (28, 28))
            im_resize = np.reshape(im_resize, -1)
            df2.loc[i][0] = label
            for k in range(img_size):
                df2.loc[i][k + 1] = im_resize[k]

        df = pd.concat([df, df2])

    df.to_csv(dataset_path + 'minst_test.csv')


def seperate_dataset(data_path):
    ratio_seperate = 0.85

    folders_list = [data_path + path for path in os.listdir(data_path) if os.path.isdir(data_path + path)]

    for folder in folders_list:
        if folder.endswith(tuple(["train", "test"])):
            print("Already Some Shit, Leaving ....")
            return None

    for folder in folders_list:
        train_folder = folder + "_train/"
        test_folder = folder + "_test/"
        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)
        folder += "/"
        file_list = [folder + im_path for im_path in os.listdir(folder) if os.path.isfile(folder + im_path)]

        for file in file_list:
            if random.random() < ratio_seperate:
                copyfile(file, train_folder + os.path.basename(file))
            else:
                copyfile(file, test_folder + os.path.basename(file))


if __name__ == '__main__':
    separate = False
    if separate:
        seperate_dataset(dataset_path)
        create_csv_seperated(dataset_path)
    else:
        create_csv(dataset_path)
