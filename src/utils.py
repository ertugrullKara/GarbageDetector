import os

import simplejson as json

try:
    with open("defaults.json", 'r') as f:
        defaults = json.load(f)
except:
    with open("../defaults.json", 'r') as f:
        defaults = json.load(f)

TRAINING_DATA_PATH = defaults["DATA_FOLDERNAME"] + os.sep + defaults["TRAINING_DATA_FILENAME"]
VALIDATION_DATA_PATH = defaults["DATA_FOLERNAME"] + os.sep + defaults["VALIDATION_DATA_FILENAME"]
TEST_DATA_PATH = defaults["DATA_FOLDERNAME"] + os.sep + defaults["TEST_DATA_FILENAME"]
CLASS_INDEX_PATH = defaults["DATA_FOLDERNAME"] + os.sep + defaults["CLASS_INDEX_FILENAME"]
CONFIG_PATH = defaults["DATA_FOLDERNAME"] + os.sep + defaults["CONFIG_FILENAME"]


def label_names_as_list():
    """
    :return: Class labels as list.
    :rtype: List
    """
    with open(CLASS_INDEX_PATH, 'r') as f:
        label_list = list(map(str.strip, f.readlines()))
    return label_list


def find_label_index(label):
    """
    :param label: Class label
    :type label: String
    :return: Index of given label.
    :rtype: Integer
    """
    with open(CLASS_INDEX_PATH, 'r') as f:
        labels = f.readlines()
    return labels.index(label)


def training_image_file_names_as_list():
    """
    :return: Training image file names as list.
    :rtype: List
    """
    with open(TRAINING_DATA_PATH, 'r') as f:
        training_list = list(map(str.strip, f.readlines()))
    return training_list


def validation_image_file_names_as_list():
    """
    :return: Validation image file names as list.
    :rtype: List
    """
    with open(VALIDATION_DATA_PATH, 'r') as f:
        validation_list = list(map(str.strip, f.readlines()))
    return validation_list


def test_image_file_names_as_list():
    """
    :return: Test image file names as list.
    :rtype: List
    """
    with open(TEST_DATA_PATH, 'r') as f:
        test_list = list(map(str.strip, f.readlines()))
    return test_list


def config_file_as_dict(config_name=CONFIG_PATH):
    """
    :param config_name: Config file name -json- to read.
    :type config_name: String
    :return: Config dictionary.
    :rtype: Dictionary
    """
    with open(config_name, 'r') as f:
        config = json.loads(f.read())
    return config


def image_normalizer(image):
    """
    :param image: Image to be normalized.
    :type image: Array
    :return: Normalized image. [0, 1]
    :rtype: Array
    """
    return image / 255.


def images_with_annots(arg=None):
    """
    Return images wrapped with annotations by labels, classes and box information.

    :param arg: Indicator of which annotations are wanted.
    :type arg: String
    :return: List of annotation Dictionaries.
    :rtype: List
    """
    imgs_annots = []
    if (arg == "train"):
        images = training_image_file_names_as_list()
    elif (arg == "validate"):
        images = validation_image_file_names_as_list()
    elif (arg == "test"):
        images = test_image_file_names_as_list()
    else:
        images = training_image_file_names_as_list() + \
                 validation_image_file_names_as_list() + \
                 test_image_file_names_as_list()

    class_names = label_names_as_list()

    for img in images:
        im = {"objects": [], "filename": img}
        ann_folder = img.rsplit('.', 1)[0] + ".txt"
        with open(ann_folder, 'r') as an:
            annots = an.readlines()
        for annot in annots:
            obj = {}
            attrs = list(map(float, annot.split()))
            obj["name"] = class_names[int(attrs[0])]
            obj["xratio"] = attrs[1]
            obj["yratio"] = attrs[2]
            obj["wratio"] = attrs[3]
            obj["hratio"] = attrs[4]
            im["objects"].append(obj)
        imgs_annots.append(im)

    return imgs_annots


if __name__ == "__main__":
    os.chdir(os.path.abspath(".."))
    print(images_with_annots())
