import os

tmp_path = "/media/hdd_linux/DataSet/0/"
images_list = [tmp_path + im_path for im_path in os.listdir(tmp_path) if os.path.isfile(tmp_path + im_path)]

images_list = sorted(images_list, key=lambda x: int(os.path.basename(x).split(".")[0]))
