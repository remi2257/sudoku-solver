import os
import pickle
import random
from shutil import copyfile

import cv2
import imgaug.augmenters as iaa
import numpy as np

list_extensions_img = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".png",
    ".webp",
]


def seperate_dataset(data_path, ratio_sep):
    folders_list = [data_path + path for path in os.listdir(data_path) if os.path.isdir(data_path + path)]

    for folder in folders_list:
        if folder.endswith(tuple(["train", "test"])):
            print("Already Some Folders, Skipping Separatation Phase ....")
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
            if random.random() < ratio_sep:
                copyfile(file, train_folder + os.path.basename(file))
            else:
                copyfile(file, test_folder + os.path.basename(file))


# --- Pickle GENERATION --- #

def create_pickle(data_root_path, list_classes, img_shape, output_name):
    training_data = []

    for classe_imgs in list_classes:
        folder_path = os.path.join(data_root_path, classe_imgs)
        class_num = list_classes.index(classe_imgs)
        for img_path in os.listdir(folder_path):
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                im_resize = cv2.resize(img, img_shape)

                training_data.append([im_resize, class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)
    X = []
    Y = []
    for img, label in training_data:
        X.append(img)
        Y.append(label)

    X = np.array(X).reshape((-1, img_shape[0], img_shape[1], 1))
    dico_pickle = {
        "X": X, "Y": Y,
        "img_shape": img_shape,
    }
    pickle_out_file = open(os.path.join(data_root_path, output_name + ".pickle"), "wb")
    pickle.dump(dico_pickle, pickle_out_file)


def read_file_classes(file_path):
    file_names = open(file_path, "r")
    raw_text = file_names.readlines()

    l_classes = [s.replace("\n", "") for s in raw_text]

    l_classes = [s for s in l_classes if s not in ["", " "]]

    return l_classes


# --- DATA AUGMENTATION --- #
def data_aug_pipeline(img):
    from matplotlib import pyplot

    seq = iaa.Sequential([
        iaa.SaltAndPepper(0.05),
        iaa.Affine(scale=(1, 1.1)),
    ])

    images_aug = []
    for i in range(9):
        image_aug = seq.augment_image(img)
        images_aug.append(image_aug)
        pyplot.subplot(3, 3, i + 1)
        pyplot.imshow(images_aug[i])
    pyplot.show()


def data_augmentation_imgaug(dataset_root_folder):
    list_folders = [entry for entry in os.scandir(dataset_root_folder) if entry.is_dir()]

    for folder in list_folders:
        list_images_path = [folder.path + "/" + entry for entry in os.listdir(folder) if
                            entry.endswith(tuple(list_extensions_img))]
        for image_path in list_images_path:
            img = cv2.imread(image_path)
            array_im_4d = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), axis=0)

            data_aug_pipeline(array_im_4d)


if __name__ == '__main__':
    dataset_path = "/media/hdd_linux/DataSet/Mine/"

    # img_path_main = dataset_path + "4/1.png"
    # im = cv2.cvtColor(cv2.imread(img_path_main), cv2.COLOR_BGR2RGB)
    # data_aug_pipeline(cv2.resize(im,(28,28)))
    #
    # ratio_seperate = 0.8
    # img_shape = (128, 128)
    names_file = os.path.join(dataset_path, "data.names")
    classes = read_file_classes(names_file)
    create_pickle(dataset_path, classes, img_shape=(28, 28), output_name="mnist_numeric")
