import os
from settings import *

class_name = str(0)
target_folder = os.path.join(my_dataset_path, class_name + "/")
print(target_folder)