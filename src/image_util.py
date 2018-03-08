# Default Libraries
import os
from random import randint, seed

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from cv2 import imshow, waitKey

# Custom Libraries
from . import utils
from .box import BoundBox


def get_random_color(class_index):
    """
    Assign a color to classes.

    :param class_index: Index of a class.
    :type class_index: Integer
    :return: Tuple of RGB values for a class.
    :rtype: Tuple
    """
    seed(class_index)
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def jpg_image_to_array(image):
    """
    Loads JPEG image into 3D Numpy array of shape
        (width, height, channels)

    :param image: JPEG image.
    :type image: Image
    :return: Array of the image.
    :rtype: Array
    """
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 4))
    return im_arr


def show_true_image_data(imgpaths=None):
    """
    Shows images with their assigned bounding boxes and classes. For manually
    checking the test results. Or just fun.

    :param imgpaths: Only shows bounding boxes for given image with paths.
    :type imgpaths: List
    """
    if imgpaths:
        training_images = list(imgpaths)
    else:
        training_images = utils.validation_image_file_names_as_list()
           
    for img in training_images:
        boxes = []
        source_img = Image.open(img).convert("RGBA")
        source_img_bbs = img.split('.')[0] + ".txt"
        
        with open(source_img_bbs, 'r') as f:
            bounding_boxes = list(map(lambda x: x.split(), f.readlines()))
            
        for bb in bounding_boxes:
            boxes.append(BoundBox(1, [100., ], bb[1:]))
            show_image(jpg_image_to_array(source_img), boxes)
            waitKey(0)
        
        # Delete or comment before calling from another directory.
        time.sleep(1)


def show_image(image, boxes, original_width=500, original_height=500):
    """
    Used to show/output image with given boxes.

    :param image: Array with shape (width, height, 3)
    :type image: Array
    :param boxes: list of boundboxes
    :type boxes: List
    :param original_width: If provided, resize the image in this sizes.
    :type original_width: Integer
    :param original_height: If provided, resize the image in this sizes.
    :type original_height: Integer

    """
    source_img = Image.fromarray(image.astype('uint8'))
    width, height = source_img.size[0], source_img.size[1]
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * height + 0.5).astype('int32'))
    thickness = (width + height) // 200

    class_list = utils.label_names_as_list()

    draw = ImageDraw.Draw(source_img)
    for box in boxes:
        class_index = box.get_classindex()
        bb_color = get_random_color(class_index)

        label = '{} {:.2f}'.format(class_list[class_index], box.class_probs[class_index])
        label_size = list(draw.textsize(label, font))

        box_x_ratio, box_y_ratio, box_width_ratio, box_height_ratio = box.get_coordinates()
        left = min(max(width * (box_x_ratio - box_width_ratio / 2), 0), width)
        top = min(max(height * (box_y_ratio - box_height_ratio / 2), 0), height)
        right = min(max(width * (box_x_ratio + box_width_ratio / 2), 0), width)
        bottom = min(max(height * (box_y_ratio + box_height_ratio / 2), 0), height)
        if top - label_size[1] >= 0:
            text_origin = [left, top - label_size[1]]
        else:
            text_origin = [left, top + 1]
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i],
                            outline=bb_color)
        draw.rectangle([tuple(text_origin), text_origin[0] + label_size[0], text_origin[1] + label_size[1]],
            fill=bb_color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    source_img = source_img.resize((original_width, original_height), Image.ANTIALIAS)
    imshow("Predicted", np.array(source_img))


if __name__ == "__main__":
    import time

    # Default running directory is parent folder of 'src' file.
    os.chdir(os.path.abspath('..'))

    show_true_image_data()
