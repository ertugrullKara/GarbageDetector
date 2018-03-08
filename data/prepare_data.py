import os
import subprocess
from glob import glob

import numpy as np
import simplejson as json


def _split_training_validation(image_list_file, validation_ratio=0.1, test_ratio=0.1):
    """
    Split data into training, validation and testing.

    :param image_list_file: Newly generated text file to split images to old ones.
    :type image_list_file: String
    :param validation_ratio: Validation data ratio for the new data.
    :type validation_ratio: Float
    :param test_ratio: Test data ratio for the new data.
    :type test_ratio: Float
    """
    images = open(image_list_file, 'r').readlines()

    np.random.shuffle(images)
    length = len(images)
    train_ratio = min(1.0, 1.0 - validation_ratio - test_ratio)

    train_split = int(train_ratio * length)
    validation_split = int(validation_ratio * length)

    tr = images[:train_split]
    val = images[train_split:train_split + validation_split]
    test = images[train_split + validation_split:]

    with open("../defaults.json", 'r') as f:
        defaults = json.load(f)

    try:
        with open(defaults["TEST_DATA_FILENAME"], 'r') as f:
            test_old = f.read().rstrip()
    except IOError:  # File does not exist yet.
        test_old = []
    try:
        with open(defaults["VALIDATION_DATA_FILENAME"], 'r') as f:
            val_old = f.read().rstrip()
    except IOError:  # File does not exist yet.
        val_old = []
    try:
        with open(defaults["TRAINING_DATA_FILENAME"], 'r') as f:
            tr_old = f.read().rstrip()
    except IOError:  # File does not exist yet.
        tr_old = []

    with open(defaults["TEST_DATA_FILENAME"], 'w') as f:
        if test_old:
            f.write(test_old + '\n')
        for img in test:
            f.write(img)
    with open(defaults["VALIDATION_DATA_FILENAME"], 'w') as f:
        if val_old:
            f.write(val_old + '\n')
        for img in val:
            f.write(img)
    with open(defaults["TRAINING_DATA_FILENAME"], 'w') as f:
        if tr_old:
            f.write(tr_old + '\n')
        for img in tr:
            f.write(img)


def _main():
    os.makedirs("raw_data/img", exist_ok=True)
    image_code = 11
    for file in glob("raw_data" + os.sep + "*.mp4"):
        image_name = "img" + repr(image_code)
        subprocess.Popen(["ffmpeg", "-i", file, "-r", "2", "raw_data/img/{}%03d.jpg".format(image_name)]).wait()
        image_code += 1
        os.remove(file)
    image_list_file = "new_images.txt"
    with open(image_list_file, 'w') as f:
        for new_jpg in glob("raw_data" + os.sep + "img" + os.sep + "*.jpg"):
            f.write("data/img/" + new_jpg.split('/')[-1] + '\n')
    _split_training_validation(image_list_file)


if __name__ == "__main__":
    _main()
    print("Done! Do not forget to run anchor generation script after labeling!")
